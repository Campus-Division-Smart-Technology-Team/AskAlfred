#!/usr/bin/env python3
"""
Document content extraction and processing for Alfred Local Ingestion.
This module provides functions to securely fetch files, extract text from various document formats,
chunk text for embedding, and handle specific cases like property and maintenance CSVs
with building name normalisation and metadata extraction.
"""

from __future__ import annotations

import io
import json
import logging
import random
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import tiktoken
from docx import Document as DocxDocument
from pypdf import PdfReader

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pylint: disable=broad-except
    pdfminer_extract_text = None
try:
    import textract  # type: ignore[import-untyped]
except Exception:  # pylint: disable=broad-except
    textract = None
from building import normalise_building_name
from config import (
    INGEST_BACKOFF_BASE,
    INGEST_BACKOFF_CAP,
    INGEST_BACKOFF_JITTER_MIN,
    INGEST_BACKOFF_JITTER_SPAN,
    INGEST_DOCX_MAX_ARCHIVE_ENTRIES,
    INGEST_DOCX_MAX_UNCOMPRESSED_MB,
    INGEST_EMBED_BATCH_SIZE,
    INGEST_FETCH_MAX_SIZE_MB,
    INGEST_PDF_MAX_PAGES,
)
from core.alfred_exceptions import ParseError, ValidationError
from domain.maintenance_utils import normalise_priority
from interfaces import EmbeddingsResult
from security.file_operations_validator import (
    ALLOWED_INGEST_EXTENSIONS,
    FileSizeError,
    FileTypeError,
    validate_file_safety,
)

if TYPE_CHECKING:
    from .context import IngestContext


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger(__name__)


_BARE_ADDRESS_NUMBER_RE = re.compile(r"^\d+(?:-\d+)?[A-Za-z]?$")
_NUMBERED_STREET_ALIAS_RE = re.compile(
    r"^(?P<number>\d+(?:-\d+)?[A-Za-z]?)\s+"
    r"(?P<street>.+\b(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|"
    r"Square|Place|Row|Terrace|Parade|Gardens)\b.*)$",
    re.IGNORECASE,
)


def _expand_compressed_address_aliases(parts: list[str]) -> list[str]:
    """Expand Planon aliases like "3, 5, 7 Woodland Road"."""
    expanded: list[str] = []
    pending_numbers: list[str] = []

    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue

        if _BARE_ADDRESS_NUMBER_RE.fullmatch(part):
            pending_numbers.append(part)
            continue

        match = _NUMBERED_STREET_ALIAS_RE.match(part)
        if match and pending_numbers:
            street = match.group("street").strip()
            expanded.extend(f"{number} {street}" for number in pending_numbers)
            pending_numbers = []
        elif pending_numbers:
            pending_numbers = []

        expanded.append(part)

    return expanded


def split_alias_field(value: str) -> list[str]:
    """
    Split a Planon alias field into individual aliases.

    Planon separates aliases with commas (occasionally semicolons), but commas
    also appear *inside* aliases within parentheses, e.g.
    "Isambard Park 1.0 (see also 1061, 1171)". Split only on top-level
    separators and drop fragments that cannot be building names (too short or
    purely numeric).
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char in ",;" and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    parts = _expand_compressed_address_aliases(parts)
    return [
        part
        for part in parts
        if len(part) >= 3 and not _BARE_ADDRESS_NUMBER_RE.fullmatch(part)
    ]


def exact_building_key(value: str | None) -> str:
    """Return a non-lossy lowercase key for exact alias/name lookups."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_pivot_header(col_str: str, is_requests: bool) -> dict[str, str]:
    """
    Parse maintenance pivot headers.

    Requests examples:
        "Asbestos Requests - Other - In progress"
        "ASU Requests - RM Priority 1 - Within 1 week - Complete"
        -> {category, priority, status}

    Jobs examples:
        "BEMS Controls Jobs - Complete"
        "ASU Jobs - In progress"
        -> {category, status}
    """
    keyword = "Requests" if is_requests else "Jobs"
    cleaned = col_str.replace(keyword, "").strip()
    parts = [p.strip() for p in re.split(r"\s*-\s*", cleaned) if p.strip()]

    if is_requests:
        if len(parts) == 1:
            return {"category": parts[0], "priority": "Unknown", "status": "Unknown"}
        if len(parts) == 2:
            return {"category": parts[0], "priority": "Unknown", "status": parts[1]}
        return {
            "category": parts[0],
            "priority": " - ".join(parts[1:-1]),
            "status": parts[-1],
        }

    if len(parts) == 1:
        return {"category": parts[0], "status": "Unknown"}
    return {"category": parts[0], "status": parts[1]}


def load_tabular_data(key: str, data: bytes) -> pd.DataFrame:
    """Load CSV or Excel data into a DataFrame based on file extension."""
    ext_ = ext(key)
    if ext_ == "csv":
        return pd.read_csv(io.BytesIO(data))
    if ext_ in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ParseError(f"Unsupported tabular file type: {ext_}")


def backoff_sleep(
    attempt: int,
    base: float = INGEST_BACKOFF_BASE,
    cap: float = INGEST_BACKOFF_CAP,
) -> None:
    """Exponential backoff with jitter."""
    delay = min(cap, base * (2**attempt)) * (
        INGEST_BACKOFF_JITTER_MIN + random.random() * INGEST_BACKOFF_JITTER_SPAN
    )
    time.sleep(delay)


def ext(key: str) -> str:
    """Extract file extension."""
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


def fetch_bytes_secure(
    base_path: str,
    key: str,
    max_size_mb: float = INGEST_FETCH_MAX_SIZE_MB,
    *,
    logger: logging.Logger | None = None,
) -> bytes:
    """
    Securely read file from local filesystem with size validation.

    Uses the file_operations_validator for comprehensive security checks.
    """
    log = _get_logger(logger)

    try:
        # Use file_operations_validator for comprehensive safety checks
        filepath = validate_file_safety(
            base_path,
            key,
            allowed_extensions=ALLOWED_INGEST_EXTENSIONS,
            max_size_mb=max_size_mb,
        )

        with open(filepath, "rb") as file_handle:
            return file_handle.read()

    except FileSizeError as e:
        log.error("File size validation failed: %s", e)
        raise ValidationError(str(e)) from e
    except FileTypeError as e:
        log.error("File type not allowed: %s", e)
        raise ParseError(str(e)) from e
    except Exception as e:
        log.error("Failed to read file %s: %s", key, e)
        raise ValidationError(f"Cannot read file: {key}") from e


def extract_text(
    key: str,
    data: bytes,
    *,
    logger: logging.Logger | None = None,
) -> str:
    """
    Extract text from standard document formats (not CSV).
    """
    extension = ext(key)

    if extension in {"txt", "md"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    if extension == "json":
        try:
            text = data.decode("utf-8")
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:  # pylint: disable=broad-except
            return data.decode("utf-8", errors="ignore")

    log = _get_logger(logger)

    if extension == "pdf":
        text = ""
        try:
            reader = PdfReader(io.BytesIO(data))
            page_count = len(reader.pages)
            if page_count > INGEST_PDF_MAX_PAGES:
                log.warning(
                    "Skipping PDF %s: %d pages exceeds limit of %d",
                    key,
                    page_count,
                    INGEST_PDF_MAX_PAGES,
                )
                return ""
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            if text.strip():
                return text
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("PDF extract failed for %s: %s", key, ex)

        if pdfminer_extract_text is None:
            return text

        if not is_fire_risk_assessment(key, text):
            return text

        try:
            log.info("Using PDFMiner fallback for %s", key)
            text = pdfminer_extract_text(io.BytesIO(data)) or ""
            return text
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("PDFMiner extract failed for %s: %s", key, ex)
            return text

    if extension == "docx":
        try:
            if not _docx_archive_within_limits(key, data, log):
                return ""
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("DOCX extract failed for %s: %s", key, ex)
            return ""

    if extension == "doc":
        return _extract_doc_text(key, data, log)

    return ""


def _docx_archive_within_limits(key: str, data: bytes, log: logging.Logger) -> bool:
    """
    Validate a DOCX zip archive before handing it to the XML parser.

    Rejects archives with excessive entry counts or declared decompressed size
    (zip bombs) so python-docx/lxml never load them.
    """
    max_uncompressed_bytes = INGEST_DOCX_MAX_UNCOMPRESSED_MB * 1024 * 1024
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > INGEST_DOCX_MAX_ARCHIVE_ENTRIES:
                log.warning(
                    "Skipping DOCX %s: %d archive entries exceeds limit of %d",
                    key,
                    len(entries),
                    INGEST_DOCX_MAX_ARCHIVE_ENTRIES,
                )
                return False
            total_uncompressed = sum(entry.file_size for entry in entries)
            if total_uncompressed > max_uncompressed_bytes:
                log.warning(
                    "Skipping DOCX %s: declared decompressed size %d bytes "
                    "exceeds limit of %d bytes",
                    key,
                    total_uncompressed,
                    max_uncompressed_bytes,
                )
                return False
    except zipfile.BadZipFile as ex:
        log.warning("Skipping DOCX %s: invalid zip archive (%s)", key, ex)
        return False
    return True


def _extract_doc_text(key: str, data: bytes, log: logging.Logger) -> str:
    if not data:
        return ""

    if textract is not None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            text_bytes = textract.process(tmp_path)
            return text_bytes.decode("utf-8", errors="ignore") if text_bytes else ""
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("DOC extract (textract) failed for %s: %s", key, ex)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    antiword_path = shutil.which("antiword")
    if antiword_path:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            res = subprocess.run(
                [antiword_path, tmp_path],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout or ""
            log.warning("DOC extract (antiword) failed for %s: %s", key, res.stderr)
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("DOC extract (antiword) failed for %s: %s", key, ex)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    log.warning("No DOC extractor available for %s", key)
    return ""


# ---------------------------------------------------------------------------
# Section-aware chunking
#
# FRAs and BMS Description-of-Operations PDFs are table-heavy template
# documents. Blind fixed-size token windows cut across template sections and
# table rows, burying key facts (e.g. an occupancy figure) in chunks whose
# embedding is dominated by unrelated text. Splitting on the documents'
# section headings keeps facts with their section, and the section title is
# prepended to each chunk so the embedding carries the topic.
# ---------------------------------------------------------------------------

# FRA template section codes, e.g. "A1", "B2", "2.1.3"-style BMS headings.
_SECTION_CODE_RE = re.compile(r"^[A-G][0-9](?:\.[0-9]{1,2})?$")
_NUMBERED_HEADING_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[A-Z][^\n]{2,70}$")
_ALL_CAPS_HEADING_RE = re.compile(
    r"^(?=[^a-z]{6,80}$)[A-Z][A-Z0-9 &/'().,:–—-]*[A-Z0-9).]$"
)
# Recurring table-column headers in the FRA templates; never section starts.
_HEADING_STOPWORDS = frozenset(
    {
        "COMMENTARY",
        "EXISTING CONTROL MEASURES",
        "FIRE RISK",
        "YES/NO",
        "DATE",
        "SIGNATURE",
        "ITEMS TO CONSIDER",
        "LOCATION",
        "DEFECT",
        "ACTION",
    }
)
_SECTION_TITLE_MAX_CHARS = 100
_SECTION_MIN_BODY_CHARS = 80
_SECTION_SPLIT_MIN_TEXT_CHARS = 2000


def _heading_title(stripped: str) -> str | None:
    """Return a section title if the line looks like a section heading."""
    if not stripped or stripped in _HEADING_STOPWORDS:
        return None
    if _SECTION_CODE_RE.match(stripped):
        return stripped
    if _ALL_CAPS_HEADING_RE.match(stripped):
        return stripped
    if _NUMBERED_HEADING_RE.match(stripped):
        return stripped
    return None


def split_text_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split extracted document text into (section_title, section_body) pairs.

    Falls back to a single untitled section when the text is short or no
    section structure is detected. Content is never dropped: tiny sections
    are folded (title included) into their predecessor.
    """
    if len(text) < _SECTION_SPLIT_MIN_TEXT_CHARS:
        return [(None, text)]

    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        title = _heading_title(stripped)
        if title is None:
            current_lines.append(lines[index])
            index += 1
            continue

        consumed = 1
        # An FRA section code ("A1") is usually followed by its caps title.
        if _SECTION_CODE_RE.match(stripped):
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines):
                follower = lines[lookahead].strip()
                if (
                    follower not in _HEADING_STOPWORDS
                    and _ALL_CAPS_HEADING_RE.match(follower)
                ):
                    title = f"{stripped} {follower}"
                    consumed = lookahead - index + 1

        sections.append((current_title, current_lines))
        current_title = title[:_SECTION_TITLE_MAX_CHARS]
        current_lines = []
        index += consumed

    sections.append((current_title, current_lines))

    # Fold sections with negligible bodies into their predecessor so that
    # stray heading-like lines do not fragment the document.
    merged: list[tuple[str | None, str]] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body and title is None:
            continue
        if len(body) < _SECTION_MIN_BODY_CHARS and merged:
            prev_title, prev_body = merged[-1]
            tail = f"{title}\n{body}" if title else body
            merged[-1] = (prev_title, f"{prev_body}\n{tail}".strip())
            continue
        merged.append((title, body))

    if len(merged) < 2:
        return [(None, text)]
    return merged


def _window_token_chunks(
    tokens: list[int],
    decode,
    chunk_tokens: int,
    chunk_overlap: int,
):
    """Yield cleaned, overlapping token-window chunks."""
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            yield chunk
        start = max(0, end - chunk_overlap)
        if end == len(tokens):
            break


def chunk_text_with_sections(
    encoder,
    text: str,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Chunk text section-by-section, prefixing each chunk with its section title.
    """
    chunks: list[str] = []
    for title, body in split_text_sections(text):
        tokens = encoder.encode(body)
        prefix = f"Section: {title}\n" if title else ""
        for chunk in _window_token_chunks(
            tokens, encoder.decode, chunk_tokens, chunk_overlap
        ):
            chunks.append(f"{prefix}{chunk}" if prefix else chunk)
    return chunks


_DOC_TYPE_HEADER_LABELS = {
    "fire_risk_assessment": "Fire Risk Assessment",
    "operational_doc": "Operational/BMS document",
    "planon_data": "Planon property data",
    "maintenance_job": "Maintenance jobs summary",
    "maintenance_request": "Maintenance requests summary",
}


def build_chunk_context_header(
    *,
    source_key: str,
    canonical: str | None = None,
    doc_type: str | None = None,
) -> str:
    """
    Build a one-line context header for a chunk.

    Embedded with (and stored alongside) every chunk of an unstructured
    document so that both retrieval and answering know which building and
    document the chunk came from.
    """
    parts: list[str] = []
    if canonical and canonical != "Unknown":
        parts.append(f"Building: {canonical}")
    if doc_type and doc_type != "unknown":
        label = _DOC_TYPE_HEADER_LABELS.get(doc_type, doc_type.replace("_", " "))
        parts.append(f"Document: {label}")
    name = Path(source_key).name if source_key else ""
    if name:
        parts.append(f"Source: {name}")
    if not parts:
        return ""
    return "[" + " | ".join(parts) + "]"


def extract_text_and_chunk_in_process(
    key: str,
    data: bytes,
    *,
    embed_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[str, list[str]]:
    """
    Extract text and chunk it (section-aware) using a local encoder.

    This is designed to run in a process pool to avoid GIL bottlenecks.
    """
    text = extract_text(key, data)
    if not text:
        return text, []

    try:
        encoder = tiktoken.encoding_for_model(embed_model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")

    chunks = chunk_text_with_sections(
        encoder,
        text,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    return text, chunks


def chunk_document_text(ctx: "IngestContext", text: str) -> list[str]:
    """Section-aware chunking for whole-document text (inline path)."""
    return chunk_text_with_sections(
        ctx.encoder,
        text,
        chunk_tokens=ctx.config.chunk_tokens,
        chunk_overlap=ctx.config.chunk_overlap,
    )


def chunk_text(ctx: "IngestContext", text: str) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    tokens = ctx.encoder.encode(text)
    chunks: list[str] = []
    start = 0
    chunk_tokens = ctx.config.chunk_tokens
    overlap = ctx.config.chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = ctx.encoder.decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == len(tokens):
            break

    return chunks


def chunk_text_generator(ctx: "IngestContext", text: str):
    """Yield overlapping chunks without storing the full chunk list."""
    tokens = ctx.encoder.encode(text)
    start = 0
    chunk_tokens = ctx.config.chunk_tokens
    overlap = ctx.config.chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = ctx.encoder.decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            yield chunk
        start = max(0, end - overlap)
        if end == len(tokens):
            break


def embed_texts_batch(
    ctx: "IngestContext",
    texts: list[str],
) -> EmbeddingsResult:
    """Embed a batch of texts; retry/backoff handled by Embedder."""
    if not texts:
        return EmbeddingsResult(
            embeddings_by_index={},
            errors_by_index={},
            error_summary=None,
        )
    if getattr(ctx.config, "dry_run", False):
        return EmbeddingsResult(
            embeddings_by_index={},
            errors_by_index={},
            error_summary=None,
        )

    t_start = time.perf_counter()
    result = ctx.embedder.embed_texts(
        texts,
        model=ctx.config.embed_model,
        timeout=ctx.config.openai_timeout,
        max_batch=INGEST_EMBED_BATCH_SIZE,
    )
    elapsed = time.perf_counter() - t_start
    ctx.stats.observe_timing("embed_batch_seconds", elapsed)
    ctx.stats.increment("embed_batches_total")
    ctx.stats.increment("embed_texts_total", len(texts))
    if result.errors_by_index:
        ctx.stats.increment("embed_failed_total", len(result.errors_by_index))
    if result.retry_attempts:
        ctx.stats.increment("embed_retries_total", result.retry_attempts)
    if result.batch_reductions:
        ctx.stats.increment("embed_batch_reductions_total", result.batch_reductions)
    if result.rate_limit_errors:
        ctx.stats.increment("embed_rate_limit_total", result.rate_limit_errors)
    return result


def load_building_names_with_aliases(
    ctx: "IngestContext",
    base_path: str,
    key: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Load building names and alias mappings from local property CSV/Excel.

    Returns:
        (canonical_names, name_to_canonical_map, alias_to_canonical_map)
    """
    existing_names = ctx.cache.get_name_mapping()
    existing_aliases = ctx.cache.get_alias_mapping()

    if existing_names and existing_aliases:
        canonical_names = list(set(existing_names.values()))
        return canonical_names, existing_names, existing_aliases

    try:
        data = fetch_bytes_secure(base_path, key)
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            ctx.logger.warning("No 'Property name' column in CSV")
            return [], {}, {}

        canonical_names: list[str] = []
        name_to_canonical: dict[str, str] = {}
        alias_to_canonical: dict[str, str] = {}
        metadata_cache: dict[str, dict[str, Any]] = {}
        duplicate_mapping_count = 0
        duplicate_mapping_examples: list[str] = []

        def record_duplicate(raw_name: str, existing: str, ignored: str) -> None:
            nonlocal duplicate_mapping_count
            duplicate_mapping_count += 1
            if len(duplicate_mapping_examples) < 5:
                duplicate_mapping_examples.append(
                    f"{raw_name}: kept {existing}, ignored {ignored}"
                )

        def add_name_mapping(raw_name: str, canonical_name: str) -> None:
            keys = [
                key
                for key in (
                    exact_building_key(raw_name),
                    normalise_building_name(raw_name),
                )
                if key
            ]
            for key in dict.fromkeys(keys):
                existing = name_to_canonical.get(key)
                if existing and existing != canonical_name:
                    record_duplicate(raw_name, existing, canonical_name)
                    continue
                name_to_canonical[key] = canonical_name

        def add_alias_mapping(raw_alias: str, canonical_name: str) -> None:
            add_name_mapping(raw_alias, canonical_name)
            keys = [
                key
                for key in (
                    exact_building_key(raw_alias),
                    normalise_building_name(raw_alias),
                )
                if key
            ]
            for key in dict.fromkeys(keys):
                existing = alias_to_canonical.get(key)
                if existing and existing != canonical_name:
                    record_duplicate(raw_alias, existing, canonical_name)
                    continue
                alias_to_canonical[key] = canonical_name

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            canonical = str(prop_name).strip()
            if not canonical:
                continue
            canonical_names.append(canonical)
            aliases = set()

            add_name_mapping(canonical, canonical)
            aliases.add(canonical)

            if pd.notna(row.get("Property names")):
                for name in split_alias_field(str(row["Property names"])):
                    add_alias_mapping(name, canonical)
                    aliases.add(name)

            if pd.notna(row.get("Property alternative names")):
                for name in split_alias_field(
                    str(row["Property alternative names"])
                ):
                    add_alias_mapping(name, canonical)
                    aliases.add(name)

            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                condensed = str(row["UsrFRACondensedPropertyName"]).strip()
                if condensed:
                    add_alias_mapping(condensed, canonical)
                    aliases.add(condensed)

            building_metadata = {
                "canonical_building_name": canonical,
                "building_aliases": list(aliases),
            }

            for field in [
                "Property code",
                "Property postcode",
                "Property campus",
                "UsrFRACondensedPropertyName",
                "Property names",
                "Property alternative names",
                "Property condition",
                "Property gross area (sq m)",
                "Property net area (sq m)",
            ]:
                if field in row.index and pd.notna(row[field]):
                    building_metadata[field] = str(row[field])

            metadata_cache[canonical] = building_metadata

        ctx.cache.update_from_csv(name_to_canonical, alias_to_canonical, metadata_cache)

        if duplicate_mapping_count:
            ctx.logger.warning(
                "Kept first canonical for %d duplicate building aliases/names. Examples: %s",
                duplicate_mapping_count,
                "; ".join(duplicate_mapping_examples),
            )

        ctx.logger.info(
            "Loaded %d canonical building names with %d total name variations",
            len(canonical_names),
            len(alias_to_canonical),
        )
        return canonical_names, name_to_canonical, alias_to_canonical

    except Exception as ex:  # pylint: disable=broad-except
        ctx.logger.warning("Failed to load building names: %s", ex)
        return [], {}, {}


def is_fire_risk_assessment(key: str, text: str = "") -> bool:
    """Determine if a document is a Fire Risk Assessment (FRA)."""
    key_lower = key.lower()
    fra_patterns = [
        r"\bfra\b",
        r"fire[\s_-]?risk[\s_-]?assessment",
        r"fire_risk",
        r"firerisk",
        r"\boas\b",
    ]

    for pattern in fra_patterns:
        if re.search(pattern, key_lower):
            return True

    if text:
        text_lower = text.lower()
        content_indicators = [
            "fire risk assessment",
            "compartmentation",
            "fire safety",
            "means of escape",
            "emergency lighting",
        ]
        count = sum(1 for indicator in content_indicators if indicator in text_lower)
        if count >= 2:
            return True

    return False


def is_bms_document(key: str) -> bool:
    """Determine if a document is related to BMS/operational systems."""
    key_lower = key.lower()
    bms_patterns = [
        r"bms",
        r"building[\s_-]?management",
        r"hvac",
        r"mechanical[\s_-]?services",
        r"electrical[\s_-]?services",
    ]
    return any(re.search(pattern, key_lower) for pattern in bms_patterns)


def extract_text_csv_by_building_enhanced(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str],
    *,
    logger: logging.Logger | None = None,
) -> list[tuple[str, str | None, str, dict[str, Any]]]:
    """
    Extract property CSV data by building.

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples.
    """
    log = _get_logger(logger)
    try:
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            log.warning(
                "Column 'Property name' not found in CSV. Available: %s",
                df.columns.tolist(),
            )
            return [
                (
                    key,
                    "",
                    data.decode("utf-8", errors="ignore"),
                    {"document_type": "unknown"},
                )
            ]

        building_docs: list[tuple[str, str | None, str, dict[str, Any]]] = []

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            raw_prop = str(prop_name).strip()
            canonical_name = alias_to_canonical.get(
                normalise_building_name(raw_prop), raw_prop
            )

            building_text = f"Property: {canonical_name}\n\n"

            for col, val in row.items():
                if col != "Property name" and pd.notna(val):
                    building_text += f"{col}: {val}\n"

            extra_metadata: dict[str, Any] = {}
            aliases = [raw_prop, canonical_name]

            if pd.notna(row.get("Property names")):
                aliases.extend(split_alias_field(str(row["Property names"])))

            if pd.notna(row.get("Property alternative names")):
                aliases.extend(
                    split_alias_field(str(row["Property alternative names"]))
                )

            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                aliases.append(str(row["UsrFRACondensedPropertyName"]).strip())
            aliases = list(set([alias for alias in aliases if alias]))

            for field in [
                "Property code",
                "Property postcode",
                "Property campus",
                "UsrFRACondensedPropertyName",
                "Property names",
                "Property alternative names",
                "Property condition",
                "Property gross area (sq m)",
                "Property net area (sq m)",
            ]:
                if field in row.index and pd.notna(row[field]):
                    extra_metadata[field] = str(row[field])

            extra_metadata["building_aliases"] = aliases
            extra_metadata["canonical_building_name"] = canonical_name
            extra_metadata["document_type"] = "planon_data"

            building_key = f"Planon Data - {canonical_name}"
            building_docs.append(
                (building_key, canonical_name, building_text, extra_metadata)
            )

        if not building_docs:
            log.warning("No buildings found in CSV, indexing as single doc")
            return [(key, "All Properties", df.to_string(), {})]

        log.info(
            "Extracted %d buildings from property CSV with aliases",
            len(building_docs),
        )
        return building_docs

    except Exception as ex:  # pylint: disable=broad-except
        log.warning("CSV extraction failed: %s", ex)
        return [
            (
                key,
                "",
                data.decode("utf-8", errors="ignore"),
                {"document_type": "unknown"},
            )
        ]


def extract_maintenance_csv(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str],
    *,
    logger: logging.Logger | None = None,
) -> list[tuple[str, str | None, str, dict[str, Any]]]:
    """
    Extract maintenance CSV data (Requests or Jobs) by building.

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples.
    """

    def _pick_building_col(df: pd.DataFrame) -> str | None:
        candidates = ["Property name", "Buildings", "Building", "Property", "Site"]
        lower_map = {column.lower().strip(): column for column in df.columns}
        for candidate in candidates:
            lower = candidate.lower().strip()
            if lower in lower_map:
                return lower_map[lower]
        return None

    log = _get_logger(logger)
    try:
        df = load_tabular_data(key, data)
        building_col = _pick_building_col(df)
        if not building_col:
            log.warning(
                "No recognised building column found in %s. Available columns: %s",
                key,
                df.columns.tolist(),
            )
            return []

        key_lower = key.lower()
        if "maintenance_jobs" in key_lower or "maintenance jobs" in key_lower:
            is_requests = False
        elif "maintenance_requests" in key_lower or "maintenance requests" in key_lower:
            is_requests = True
        else:
            has_requests = any(
                "requests" in str(col).lower()
                for col in df.columns
                if col != building_col
            )
            has_jobs = any(
                "jobs" in str(col).lower() for col in df.columns if col != building_col
            )
            is_requests = has_requests and not has_jobs

        data_type = "Maintenance Requests" if is_requests else "Maintenance Jobs"
        log.info("Processing %s with %d rows", data_type, len(df))

        # Pivot headers are identical for every row, so parse and normalise
        # them once up front instead of once per cell.
        parsed_headers: dict[Any, dict[str, str]] = {}
        for col in df.columns:
            if col == building_col:
                continue
            col_str = str(col)
            col_lower = col_str.lower()
            if is_requests:
                if "request" not in col_lower:
                    continue
            elif not any(token in col_lower for token in ("job", "work order")):
                continue

            try:
                parsed = parse_pivot_header(col_str, is_requests)
                status = (str(parsed.get("status") or "").strip()) or "unknown"
                parsed["status"] = status[:1].upper() + status[1:].lower()
                parsed["priority"] = normalise_priority(parsed.get("priority") or "")
                parsed_headers[col] = parsed
            except Exception as error:
                log.warning(
                    "Could not parse column header '%s': %s",
                    col,
                    error,
                )

        building_docs: list[tuple[str, str | None, str, dict[str, Any]]] = []

        for _, row in df.iterrows():
            building_name = row.get(building_col)
            if pd.isna(building_name):
                continue

            raw = str(building_name).strip()
            if not raw:
                continue
            canonical_name = alias_to_canonical.get(normalise_building_name(raw), raw)

            text_parts = [
                f"Building: {canonical_name}",
                f"Data Type: {data_type}",
                "",
            ]

            doc_type = "maintenance_request" if is_requests else "maintenance_job"
            extra_metadata: dict[str, Any] = {
                "canonical_building_name": canonical_name,
                "document_type": doc_type,
                "data_type": data_type.lower().replace(" ", "_"),
                "file_source": key,
            }

            maintenance_metrics: dict[str, Any] = {}
            total_items = 0

            for col, parsed in parsed_headers.items():
                val = row[col]
                if pd.isna(val) or val == 0:
                    continue

                try:
                    category = parsed["category"]
                    status = parsed["status"]
                    priority_label = parsed["priority"]
                    count = int(float(val))

                    if is_requests:
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category].setdefault(priority_label, {})
                        leaf = maintenance_metrics[category][priority_label]
                        leaf[status] = leaf.get(status, 0) + count
                        text_parts.append(
                            f"{category} - {priority_label} - {status}: {count}"
                        )
                    else:
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category][status] = (
                            maintenance_metrics[category].get(status, 0) + count
                        )
                        text_parts.append(f"{category} - {status}: {count}")

                    total_items += count
                except Exception as error:
                    log.warning(
                        "Could not parse column '%s' with value '%s': %s",
                        col,
                        val,
                        error,
                    )

            if not maintenance_metrics:
                log.warning(
                    "No maintenance metrics extracted for %s (%s).",
                    canonical_name,
                    data_type,
                )

            metrics_json = json.dumps(maintenance_metrics)
            if len(metrics_json) > 30000:
                log.warning(
                    "Maintenance metrics too large for %s (%d bytes), truncating",
                    canonical_name,
                    len(metrics_json),
                )
                extra_metadata["maintenance_metrics"] = json.dumps(
                    {
                        "total_items": total_items,
                        "categories_count": len(maintenance_metrics),
                        "categories": list(maintenance_metrics.keys()),
                    }
                )
                text_parts.append("")
                text_parts.append("=== Summary ===")
                text_parts.append(f"Total {data_type}: {total_items}")
                text_parts.append(
                    f"Categories with activity: {len(maintenance_metrics)}"
                )
                text_parts.append(
                    "Active categories: "
                    + ", ".join(sorted(maintenance_metrics.keys()))
                )
            else:
                extra_metadata["maintenance_metrics"] = metrics_json

            building_text = "\n".join(text_parts) + "\n"
            doc_key = f"{data_type} - {canonical_name}"
            building_docs.append(
                (doc_key, canonical_name, building_text, extra_metadata)
            )

        log.info("Extracted %d buildings from %s", len(building_docs), key)
        return building_docs

    except Exception as ex:  # pylint: disable=broad-except
        log.error(
            "Maintenance CSV extraction failed for %s: %s",
            key,
            ex,
            exc_info=True,
        )
        return []
