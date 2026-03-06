#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Inventory Summarizer

Scans a directory and uses the OpenAI client to generate a summary for each file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections.abc import Iterable
from pathlib import Path
from typing import Optional
import fnmatch

from clients import get_oai
from ingest.document_content import extract_text
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
from config.constant import (
    ANSWER_MODEL,
    INGEST_BACKOFF_BASE,
    INGEST_BACKOFF_CAP,
    INGEST_BACKOFF_JITTER_MIN,
    INGEST_BACKOFF_JITTER_SPAN,
    INGEST_RETRY_EXP_MAX,
    INGEST_RETRY_EXP_MIN,
    INGEST_RETRY_EXP_MULTIPLIER,
    INGEST_RETRY_ATTEMPTS,
)
from building.path_inventory import PathEntry, scan_path


DEFAULT_EXTENSIONS = {
    "txt", "md", "rst", "json", "csv", "tsv",
    "yaml", "yml", "ini", "cfg", "conf", "log",
    "py", "js", "ts", "html", "css",
    "pdf", "doc", "docx",
}

# Enhanced sensitive patterns with case-insensitive and variant detection
SENSITIVE_PATTERNS = {
    ".env", ".env.local", ".env.*.local",
    "*.pem", "*.key", "*.cert", "*.crt",
    ".aws", ".ssh", ".git/config",
    "secrets.json", "credentials.json",
    ".npmrc", ".netrc", "docker.env",
    "*.keystore", "*.jks", "*.p12", "*.pfx",
    "*password*", "*api*key*", "*token*", "*secret*",
}

# Regex-based sensitive content detection (best-effort, conservative)
SENSITIVE_CONTENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private_key_block", r"-----BEGIN (RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----"),
    ("aws_access_key_id", r"AKIA[0-9A-Z]{16}"),
    ("aws_secret_access_key",
     r"(?i)aws(.{0,20})?secret(.{0,20})?key\s*[:=]\s*[A-Za-z0-9/+=]{40}"),
    ("github_pat", r"ghp_[A-Za-z0-9]{36}"),
    ("github_fine_grained_pat", r"github_pat_[A-Za-z0-9_]{82,}"),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,48}"),
    ("google_api_key", r"AIza[0-9A-Za-z\-_]{35}"),
    ("jwt", r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    ("generic_password", r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?.{8,}"),
    ("generic_token",
     r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
)

logger = logging.getLogger(__name__)


class _NoTextExtractedError(ValueError):
    """Raised when a binary document yields no extractable text."""


def summarize_path(
    root_path: str,
    *,
    model: str = ANSWER_MODEL,
    max_depth: Optional[int] = None,
    max_bytes: int = 40_000_000,
    max_chars: int = 100_000,
    extensions: Optional[set[str]] = None,
    follow_symlinks: bool = False,
    on_error: str = "skip",
    workers: int = 4,
    stream: bool = False,
    output_path: str = "-",
) -> dict:
    """
    Scan root_path and summarize each file with the OpenAI client.
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS

    # Validate and normalize root_path
    try:
        root_resolved = Path(root_path).resolve()
        if not root_resolved.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        if not root_resolved.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid root_path: {exc}") from exc

    tree = scan_path(
        root_path,
        include_files=True,
        include_dirs=True,
        max_depth=max_depth,
        use_utc=True,
        follow_symlinks=follow_symlinks,
        on_error=on_error,
    )
    files = [e for e in _iter_files(tree)]

    summaries: list[dict] = []
    stream_writer = _StreamWriter(enabled=stream, output_path=output_path)
    lock = threading.Lock()
    if workers < 1:
        workers = 1

    def _process(entry: PathEntry) -> dict:
        path = Path(entry.path)

        # Verify path is within root_resolved
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            return _build_result(entry, None, "skipped_outside_root")

        if not _is_allowed_extension(path, extensions):
            return _build_result(entry, None, "skipped_extension")

        if _is_sensitive_file(path):
            return _build_result(entry, None, "skipped_sensitive")

        if _is_binary_file(path) and not _is_text_extractable_binary(path):
            return _build_result(entry, None, "skipped_binary")

        try:
            content = _read_text(
                path,
                root_resolved=root_resolved,
                max_bytes=max_bytes,
                max_chars=max_chars,
            )
            if not content.strip():
                return _build_result(entry, "Empty file.", None)
            if _has_sensitive_content(content):
                return _build_result(entry, None, "skipped_sensitive_content")

            client = get_oai()
            summary = _summarize_text(
                client=client,
                model=model,
                filename=entry.name,
                content=content,
            )
            return _build_result(entry, summary, None)
        except _NoTextExtractedError as exc:
            logger.warning("No extractable text: %s", exc)
            return _build_result(entry, None, "skipped_no_text")
        except Exception:  # pylint: disable=broad-except
            logger.exception("Error processing file: %s", entry.path)
            return _build_result(entry, None, "error_processing_file")

    if workers <= 1:
        for entry in files:
            result = _process(entry)
            summaries.append(result)
            stream_writer.write(result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process, e) for e in files]
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    summaries.append(result)
                stream_writer.write(result)

    return {
        "root_path": str(root_resolved),
        "generated_at": _utc_now_iso(),
        "model": model,
        "max_bytes": max_bytes,
        "max_chars": max_chars,
        "extensions": sorted(extensions),
        "files": summaries,
    }


def _iter_files(entry: PathEntry) -> Iterable[PathEntry]:
    for child in entry.children:
        if child.entry_type == "file":
            yield child
        if child.children:
            yield from _iter_files(child)


def _is_allowed_extension(path: Path, extensions: set[str]) -> bool:
    if not path.suffix:
        return False
    return path.suffix.lstrip(".").lower() in extensions


def _is_sensitive_file(path: Path) -> bool:
    """Check if file matches sensitive patterns (case-insensitive)."""
    name_lower = path.name.lower()
    parts_lower = [p.lower() for p in path.parts]

    for pattern in SENSITIVE_PATTERNS:
        pattern_lower = pattern.lower()

        # Exact match
        if pattern_lower.replace("*", "") == name_lower:
            return True

        # Wildcard extensions
        if pattern.startswith("*."):
            ext = pattern[1:].lower()
            if name_lower.endswith(ext):
                return True

        # Wildcard in filename (e.g., *password*)
        if "*" in pattern and "/" not in pattern:
            if fnmatch.fnmatch(name_lower, pattern_lower):
                return True

        # Directory-based match
        if "/" in pattern:
            parts = pattern_lower.split("/")
            if all(p in parts_lower for p in parts):
                return True

    return False


def _is_binary_file(path: Path) -> bool:
    """Enhanced binary file detection."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(8192)  # Increased chunk size
        # Check for null bytes (strong binary indicator)
        if b"\x00" in chunk:
            return True
        # Check for high proportion of non-text bytes
        if len(chunk) > 0:
            non_text = sum(1 for b in chunk if b <
                           0x20 and b not in (0x09, 0x0A, 0x0D))
            if non_text / len(chunk) > 0.3:
                return True
        return False
    except Exception:
        return True  # Skip on any error


def _is_text_extractable_binary(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".docx", ".doc"}


def _has_sensitive_content(content: str) -> bool:
    """Best-effort detection of secrets in content."""
    for name, pattern in SENSITIVE_CONTENT_PATTERNS:
        try:
            if re.search(pattern, content):
                logger.warning(
                    "Sensitive content detected (%s); skipping file.", name)
                return True
        except re.error:
            logger.warning("Invalid sensitive content regex: %s", name)
    return False


def _read_text(
    path: Path,
    *,
    root_resolved: Path,
    max_bytes: int,
    max_chars: int,
) -> str:
    """Enhanced text reading with best-effort open-by-handle validation."""
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(root_resolved)

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(resolved_path, flags)
        except TypeError:
            fd = os.open(resolved_path, os.O_RDONLY)

        try:
            stat_fd = os.fstat(fd)
            stat_path = os.stat(resolved_path)
            if not _same_file_stats(stat_fd, stat_path):
                raise ValueError(f"File changed during open: {resolved_path}")

            with os.fdopen(fd, "rb") as handle:
                ext = path.suffix.lower().lstrip(".")
                if ext in {"pdf", "docx", "doc"}:
                    raw = handle.read()
                    text = extract_text(path.name, raw, logger=logger)
                    if not (text or "").strip():
                        raise _NoTextExtractedError(
                            f"PDF/DOCX extraction yielded no text for: {path}"
                        )
                    return text[:max_chars]
                raw = handle.read(max_bytes)
        finally:
            if fd >= 0:  # Only close valid file descriptors
                try:
                    os.close(fd)
                except Exception:
                    pass

        # Strict mode -- fail on invalid UTF-8
        text = raw.decode("utf-8", errors="strict")
        return text[:max_chars]
    except UnicodeDecodeError as exc:
        logger.warning("File is not valid UTF-8: %s", path)
        raise ValueError(f"File contains invalid UTF-8: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read file safely: {path}") from exc


def _same_file_stats(a: os.stat_result, b: os.stat_result) -> bool:
    """Best-effort same-file check using stat results."""
    try:
        if a.st_ino and b.st_ino:
            return a.st_ino == b.st_ino and a.st_dev == b.st_dev
    except Exception:
        pass
    return (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)


def _summarize_text(*, client, model: str, filename: str, content: str) -> str:
    """Summarize text with retry logic for rate limiting."""
    prompt = (
        "Summarize the following file content in 1-3 sentences. "
        "Be concise and factual.\n\n"
        f"Filename: {filename}\n\n"
        f"Content:\n{content}"
    )

    max_retries = INGEST_RETRY_ATTEMPTS

    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise file summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            msg = response.choices[0].message.content if response.choices else ""
            return (msg or "").strip()
        except RateLimitError as exc:
            last_error = type(exc).__name__
            if attempt < max_retries - 1:
                exp = min(
                    INGEST_RETRY_EXP_MAX,
                    INGEST_RETRY_EXP_MIN *
                    (INGEST_RETRY_EXP_MULTIPLIER ** attempt),
                )
                base = min(INGEST_BACKOFF_CAP, max(INGEST_BACKOFF_BASE, exp))
                jitter = INGEST_BACKOFF_JITTER_MIN + random.random() * INGEST_BACKOFF_JITTER_SPAN
                sleep_s = min(INGEST_BACKOFF_CAP, base + jitter)
                logger.warning("Rate limited (attempt %d/%d), retrying in %.2fs",
                               attempt + 1, max_retries, sleep_s)
                time.sleep(sleep_s)
            else:
                logger.exception(
                    "Rate limited after %d attempts", max_retries)
                return (
                    "Error: Rate limited after multiple retry attempts "
                    f"(last_error={last_error})."
                )
        except (APITimeoutError, APIConnectionError) as exc:
            last_error = type(exc).__name__
            if attempt < max_retries - 1:
                exp = min(
                    INGEST_RETRY_EXP_MAX,
                    INGEST_RETRY_EXP_MIN *
                    (INGEST_RETRY_EXP_MULTIPLIER ** attempt),
                )
                base = min(INGEST_BACKOFF_CAP, max(INGEST_BACKOFF_BASE, exp))
                jitter = INGEST_BACKOFF_JITTER_MIN + random.random() * INGEST_BACKOFF_JITTER_SPAN
                sleep_s = min(INGEST_BACKOFF_CAP, base + jitter)
                logger.warning("API timeout/connection error (attempt %d/%d), retrying in %.2fs: %s",
                               attempt + 1, max_retries, sleep_s, type(exc).__name__)
                time.sleep(sleep_s)
            else:
                logger.exception(
                    "API timeout/connection error after %d attempts", max_retries)
                return (
                    "Error: API timeout/connection error after multiple retry attempts "
                    f"(last_error={last_error})."
                )
        except (AuthenticationError, PermissionDeniedError) as exc:
            last_error = type(exc).__name__
            logger.exception(
                "Authentication/permission error; not retrying: %s", last_error)
            return (
                "Error: Authentication/permission error; request not retried "
                f"(last_error={last_error})."
            )
        except Exception as exc:
            last_error = type(exc).__name__
            if attempt < max_retries - 1:
                exp = min(
                    INGEST_RETRY_EXP_MAX,
                    INGEST_RETRY_EXP_MIN *
                    (INGEST_RETRY_EXP_MULTIPLIER ** attempt),
                )
                base = min(INGEST_BACKOFF_CAP, max(INGEST_BACKOFF_BASE, exp))
                jitter = INGEST_BACKOFF_JITTER_MIN + random.random() * INGEST_BACKOFF_JITTER_SPAN
                sleep_s = min(INGEST_BACKOFF_CAP, base + jitter)
                logger.warning("API request failed (attempt %d/%d), retrying in %.2fs: %s",
                               attempt + 1, max_retries, sleep_s, last_error)
                time.sleep(sleep_s)
            else:
                logger.exception(
                    "API request failed after %d attempts", max_retries)
                return (
                    "Error: Unable to summarize file after multiple retry attempts "
                    f"(last_error={last_error})."
                )

    return (
        "Error: Unable to summarize file after multiple retry attempts "
        f"(last_error={last_error or 'Unknown'})."
    )


def _build_result(entry: PathEntry, summary: Optional[str], status: Optional[str]) -> dict:
    return {
        "path": entry.path,
        "relative_path": entry.relative_path,
        "last_write_time": entry.last_write_time,
        "size_bytes": entry.size_bytes,
        "summary": summary,
        "status": status,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a path and summarize each file using OpenAI.",
    )
    parser.add_argument(
        "root_path",
        help="Root directory to scan (UNC/network paths supported on Windows).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output file path for JSON. Use '-' for stdout (default).",
    )
    parser.add_argument(
        "--model",
        default=ANSWER_MODEL,
        help=f"OpenAI model to use (default: {ANSWER_MODEL}).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to traverse (0 = root only). Default: unlimited.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=4_000_000,
        help="Maximum bytes read from each file (default: 4000000).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100_000,
        help="Maximum characters sent to the model per file (default: 100000).",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated list of allowed file extensions.",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        default=False,
        help="Follow symlinks (default: skip).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel summarization (default: 4).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream results as JSON Lines while processing.",
    )
    parser.add_argument(
        "--on-error",
        choices=["skip", "raise"],
        default="skip",
        help="Behavior on error while scanning (default: skip).",
    )
    return parser


def _parse_extensions(raw: str) -> set[str]:
    return {ext.strip().lstrip(".").lower() for ext in raw.split(",") if ext.strip()}


class _StreamWriter:
    """Helper class to write streaming output safely from multiple threads."""

    def __init__(self, *, enabled: bool, output_path: str) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self._lock = threading.Lock()

    def write(self, item: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(item, ensure_ascii=True)
        with self._lock:
            if self.output_path == "-":
                print(line)
            else:
                with self._lock:
                    with Path(self.output_path).open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")


def _write_output(payload: dict, output_path: str) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=True)
    if output_path == "-":
        print(data)
        return
    Path(output_path).write_text(data, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    extensions = _parse_extensions(args.extensions)

    logger.info("Starting path inventory summarization for: %s", args.root_path)
    logger.info("WARNING: File content will be sent to OpenAI API for summarization. "
                "Ensure no sensitive data is exposed.")

    try:
        payload = summarize_path(
            args.root_path,
            model=args.model,
            max_depth=args.max_depth,
            max_bytes=args.max_bytes,
            max_chars=args.max_chars,
            extensions=extensions,
            follow_symlinks=args.follow_symlinks,
            on_error=args.on_error,
            workers=args.workers,
            stream=args.stream,
            output_path=args.output,
        )
        if args.stream and args.output != "-":
            _write_output(payload, args.output)
        elif not args.stream:
            _write_output(payload, args.output)

        logger.info("Summarization complete. Processed %d files.",
                    len(payload["files"]))
        return 0
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during summarization")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
