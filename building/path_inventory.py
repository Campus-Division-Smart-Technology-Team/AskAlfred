#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Inventory Scanner

Traverse a filesystem path (including UNC/network shares on Windows) and return
structured output describing subfolders, files, and LastWriteTime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import argparse
import json
import sys


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class PathEntry:
    path: str
    relative_path: str
    name: str
    entry_type: str  # "dir" or "file"
    last_write_time: str
    size_bytes: Optional[int] = None
    children: list["PathEntry"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return asdict(self)


# =============================================================================
# CORE SCAN LOGIC
# =============================================================================

def scan_path(
    root_path: str,
    *,
    include_files: bool = True,
    include_dirs: bool = True,
    max_depth: Optional[int] = None,
    use_utc: bool = True,
    follow_symlinks: bool = False,
    on_error: str = "skip",  # "skip" or "raise"
) -> PathEntry:
    """
    Scan a path and return a structured tree of folders and files.

    Args:
        root_path: Path to scan (UNC/network paths supported on Windows).
        include_files: Include files in output.
        include_dirs: Include directories in output.
        max_depth: Maximum depth to traverse (0 = root only). None = unlimited.
        use_utc: If True, LastWriteTime is ISO-8601 UTC (Z). Otherwise local time.
        follow_symlinks: If True, follow symlinks. If False, skip them.
        on_error: "skip" to ignore unreadable entries, "raise" to fail fast.

    Returns:
        PathEntry for the root directory.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")

    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    return _scan_dir(
        root=root,
        current=root,
        depth=0,
        include_files=include_files,
        include_dirs=include_dirs,
        max_depth=max_depth,
        use_utc=use_utc,
        follow_symlinks=follow_symlinks,
        on_error=on_error,
    )


def _scan_dir(
    *,
    root: Path,
    current: Path,
    depth: int,
    include_files: bool,
    include_dirs: bool,
    max_depth: Optional[int],
    use_utc: bool,
    follow_symlinks: bool,
    on_error: str,
) -> PathEntry:
    entry = _build_entry(
        root=root,
        target=current,
        entry_type="dir",
        use_utc=use_utc,
    )

    if max_depth is not None and depth >= max_depth:
        return entry

    try:
        children = sorted(
            current.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except Exception:
        if on_error == "raise":
            raise
        return entry

    for child in children:
        try:
            if child.is_symlink() and not follow_symlinks:
                continue

            if child.is_dir():
                if include_dirs:
                    entry.children.append(
                        _scan_dir(
                            root=root,
                            current=child,
                            depth=depth + 1,
                            include_files=include_files,
                            include_dirs=include_dirs,
                            max_depth=max_depth,
                            use_utc=use_utc,
                            follow_symlinks=follow_symlinks,
                            on_error=on_error,
                        )
                    )
                elif max_depth is None or depth + 1 <= max_depth:
                    # Still descend to find files if dirs are excluded
                    entry.children.extend(
                        _scan_dir(
                            root=root,
                            current=child,
                            depth=depth + 1,
                            include_files=include_files,
                            include_dirs=include_dirs,
                            max_depth=max_depth,
                            use_utc=use_utc,
                            follow_symlinks=follow_symlinks,
                            on_error=on_error,
                        ).children
                    )
            elif child.is_file():
                if include_files:
                    entry.children.append(
                        _build_entry(
                            root=root,
                            target=child,
                            entry_type="file",
                            use_utc=use_utc,
                        )
                    )
        except Exception:
            if on_error == "raise":
                raise
            continue

    return entry


def _build_entry(*, root: Path, target: Path, entry_type: str, use_utc: bool) -> PathEntry:
    stat = target.stat()
    last_write_time = _to_iso(stat.st_mtime, use_utc=use_utc)
    relative_path = str(target.relative_to(root))
    if relative_path == ".":
        relative_path = ""

    size_bytes = stat.st_size if entry_type == "file" else None
    return PathEntry(
        path=str(target),
        relative_path=relative_path,
        name=target.name,
        entry_type=entry_type,
        last_write_time=last_write_time,
        size_bytes=size_bytes,
    )


def _to_iso(timestamp: float, *, use_utc: bool) -> str:
    if use_utc:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return datetime.fromtimestamp(timestamp).isoformat()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a path and output a JSON inventory of folders/files.",
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
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to traverse (0 = root only). Default: unlimited.",
    )
    parser.add_argument(
        "--include-files",
        action="store_true",
        default=True,
        help="Include files in output (default).",
    )
    parser.add_argument(
        "--exclude-files",
        action="store_true",
        default=False,
        help="Exclude files from output.",
    )
    parser.add_argument(
        "--include-dirs",
        action="store_true",
        default=True,
        help="Include directories in output (default).",
    )
    parser.add_argument(
        "--exclude-dirs",
        action="store_true",
        default=False,
        help="Exclude directories from output.",
    )
    parser.add_argument(
        "--local-time",
        action="store_true",
        default=False,
        help="Use local time for LastWriteTime (default is UTC).",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        default=False,
        help="Follow symlinks (default: skip).",
    )
    parser.add_argument(
        "--on-error",
        choices=["skip", "raise"],
        default="skip",
        help="Behavior on error while scanning (default: skip).",
    )
    return parser


def _resolve_includes(args: argparse.Namespace) -> tuple[bool, bool]:
    include_files = args.include_files and not args.exclude_files
    include_dirs = args.include_dirs and not args.exclude_dirs
    return include_files, include_dirs


def _write_output(payload: dict, output_path: str) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=True)
    if output_path == "-":
        sys.stdout.write(data + "\n")
        return
    Path(output_path).write_text(data, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    include_files, include_dirs = _resolve_includes(args)

    tree = scan_path(
        args.root_path,
        include_files=include_files,
        include_dirs=include_dirs,
        max_depth=args.max_depth,
        use_utc=not args.local_time,
        follow_symlinks=args.follow_symlinks,
        on_error=args.on_error,
    )
    _write_output(tree.to_dict(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
