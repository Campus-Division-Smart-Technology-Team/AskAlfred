#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-safe ingestion utilities and metadata validation for AskAlfred.
Fixed version: adds complete ThreadSafeVectorBuffer and validate_namespace_routing.
"""

import json
import logging
import time
import pickle
import hashlib
from threading import Lock, RLock
from typing import Dict, Any, Optional, Set, Tuple, List
from pathlib import Path
from config import resolve_namespace


# ---------------------------------------------------------------------------
# INGEST CONTEXT
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ingest_context import IngestContext

# ---------------------------------------------------------------------------
# PATH HANDLING
# ---------------------------------------------------------------------------


def validate_safe_path(base_path: str, key: str) -> Path:
    """
    Validate that a file key is safe and within the base directory.
    Prevents path traversal attacks.

    Args:
        base_path: Base directory path
        key: Relative file path/key

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is unsafe or outside base directory
    """
    base = Path(base_path).resolve()
    target = (base / key).resolve()

    # Check if target is within base directory
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"Path traversal detected: '{key}' resolves outside base directory"
        ) from exc

    # Reject symlinks explicitly
    if target.is_symlink():
        raise ValueError(f"Symlinks are not allowed: {key}")
    # Additional security checks
    if not target.exists():
        raise FileNotFoundError(f"File not found: {key}")

    if not target.is_file():
        raise ValueError(f"Not a regular file: {key}")

    # Check for suspicious patterns
    suspicious_patterns = ['..', '~', '$']
    for pattern in suspicious_patterns:
        if pattern in str(key):
            logging.warning(
                "Suspicious pattern '%s' found in path: %s",
                pattern, key
            )

    return target

# ---------------------------------------------------------------------------
# THREAD-SAFE STATS
# ---------------------------------------------------------------------------


class ThreadSafeStats:
    def __init__(self):
        self._lock = Lock()
        self._stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "total_vectors": 0,
            "vectors_skipped": 0,
            "failed_files": [],
        }

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    def append_failed(self, filename: str) -> None:
        with self._lock:
            self._stats["failed_files"].append(filename)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._stats.copy()


# ---------------------------------------------------------------------------
# THREAD-SAFE CACHE
# ---------------------------------------------------------------------------

class ThreadSafeCache:
    def __init__(self):
        self._lock = RLock()
        self._name_cache: Dict[str, str] = {}
        self._alias_cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    def update_from_csv(
        self,
        name_to_canonical: Dict[str, str],
        alias_to_canonical: Dict[str, str],
        metadata_cache: Dict[str, Dict[str, Any]],
    ) -> None:
        with self._lock:
            self._name_cache.update(name_to_canonical)
            self._alias_cache.update(alias_to_canonical)
            for k, v in metadata_cache.items():
                self._metadata_cache[k] = v.copy()

    def get_name_mapping(self) -> Dict[str, str]:
        with self._lock:
            return self._name_cache.copy()

    def get_alias_mapping(self) -> Dict[str, str]:
        with self._lock:
            return self._alias_cache.copy()

    def get_metadata(self, building_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached metadata for a building.
        Fixed: Added missing implementation.

        Args:
            building_name: The canonical building name

        Returns:
            Dictionary of cached metadata or None if not found
        """
        with self._lock:
            metadata = self._metadata_cache.get(building_name)
            # Return a copy to prevent external modifications
            return metadata.copy() if metadata else None

    def set_metadata(self, building_name: str, metadata: Dict[str, Any]) -> None:
        """
        Cache metadata for a building.

        Args:
            building_name: The canonical building name
            metadata: Metadata dictionary to cache
        """
        with self._lock:
            self._metadata_cache[building_name] = metadata.copy()

    def has_metadata(self, building_name: str) -> bool:
        """Check if metadata exists for a building."""
        with self._lock:
            return building_name in self._metadata_cache


# ---------------------------------------------------------------------------
# VECTOR ID CACHE
# ---------------------------------------------------------------------------

class VectorIDCache:
    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()

    def _get_cache_path(self, source_path: str, namespace: Optional[str]) -> Path:
        ns = namespace or "default"
        h = hashlib.md5(source_path.encode()).hexdigest()
        return self.cache_dir / f"{ns}_{h}.pkl"

    def get(self, source_path: str, namespace: Optional[str]) -> Optional[Set[str]]:
        file = self._get_cache_path(source_path, namespace)
        with self._lock:
            if not file.exists():
                return None
            if time.time() - file.stat().st_mtime > self.ttl_seconds:
                return None
            try:
                with open(file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None

    def set(self, source_path: str, namespace: Optional[str], ids: Set[str]) -> None:
        file = self._get_cache_path(source_path, namespace)
        with self._lock:
            with open(file, "wb") as f:
                pickle.dump(ids, f)

    def clear_all(self) -> None:
        """Clear all cached vector ID files."""
        with self._lock:
            if not self.cache_dir.exists():
                return
            try:
                for file in self.cache_dir.glob("*.pkl"):
                    file.unlink(missing_ok=True)
                logging.info(
                    "🧹 Cleared all vector ID cache files from %s", self.cache_dir)
            except Exception as e:
                logging.warning("Failed to clear vector ID cache: %s", e)


# ---------------------------------------------------------------------------
# METADATA VALIDATOR
# ---------------------------------------------------------------------------
def truncate_by_tokens(text: str, encoder, max_tokens: int) -> str:
    if not text:
        return text
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


class MetadataValidator:
    REQUIRED_FIELDS = {"source_path", "key", "source", "text", "document_type"}
    MAX_METADATA_SIZE = 40960
    MAX_TEXT_TOKENS = 1000

    @staticmethod
    def validate(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        missing = MetadataValidator.REQUIRED_FIELDS - metadata.keys()
        if missing:
            return False, f"Missing required fields: {missing}"

        nulls = [k for k, v in metadata.items() if v is None]
        if nulls:
            return False, f"Null values in {nulls}"

        try:
            size = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
            if size > MetadataValidator.MAX_METADATA_SIZE:
                return False, "Metadata too large"
        except Exception as e:
            return False, f"Serialisation error: {e}"

        return True, None


# ---------------------------------------------------------------------------
# THREAD-SAFE VECTOR BUFFER
# ---------------------------------------------------------------------------

class ThreadSafeVectorBuffer:
    """Thread-safe buffer for pending vectors."""

    def __init__(self, max_size: int = 10000):
        self._lock = RLock()
        self._buffer: List[Dict[str, Any]] = []
        self.max_size = max_size

    def add(self, vector: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._buffer) >= self.max_size:
                raise BufferError("Vector buffer full")
            self._buffer.append(vector)

    def get_and_clear(self) -> List[Dict[str, Any]]:
        """Retrieve and clear all buffered vectors (used for upserts)."""
        with self._lock:
            data = self._buffer[:]
            self._buffer.clear()
            return data

    def extend(self, vectors: List[Dict[str, Any]]) -> None:
        """Add multiple vectors to the buffer."""
        with self._lock:
            remaining_space = self.max_size - len(self._buffer)
            if len(vectors) > remaining_space:
                raise BufferError("Not enough space in buffer for extend()")
            self._buffer.extend(vectors)

    def size(self) -> int:
        """Return the current size of the buffer."""
        with self._lock:
            return len(self._buffer)

    def __len__(self) -> int:
        """Return the number of vectors in the buffer (len() support)."""
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        """Return True if the buffer has no vectors."""
        with self._lock:
            return len(self._buffer) == 0


# ---------------------------------------------------------------------------
# VALIDATE NAMESPACE ROUTING
# ---------------------------------------------------------------------------

def validate_namespace_routing(
    doc_type: Optional[str],
    resolved_namespace: Optional[str],
) -> tuple[bool, str]:
    try:
        expected = resolve_namespace(doc_type)
    except ValueError as e:
        return False, str(e)

    if resolved_namespace != expected:
        return False, f"Namespace mismatch: expected '{expected}' for doc_type '{doc_type}', got '{resolved_namespace}'"

    return True, "ok"

# ============================================================================
# PROGRESS BAR IMPLEMENTATION
# ============================================================================


class IngestionProgressTracker:
    """
    Tracks and displays ingestion progress with detailed statistics.
    Supports both sequential and parallel processing modes.
    """

    def __init__(self, total_files: int, use_tqdm: bool = True):
        """
        InitialiSe progress tracker.

        Args:
            total_files: Total number of files to process
            use_tqdm: Whether to use tqdm progress bar (requires package)
        """
        self.total_files = total_files
        self.use_tqdm = use_tqdm
        self.pbar = None

        # Statistics
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.vectors_created = 0

        if use_tqdm:
            try:
                from tqdm import tqdm
                self.pbar = tqdm(
                    total=total_files,
                    desc="Processing files",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            except ImportError:
                logging.warning(
                    "tqdm not installed, falling back to simple progress logging. "
                    "Install with: pip install tqdm"
                )
                self.use_tqdm = False

    def update(self, filename: str, vectors: int = 0, status: str = "processed"):
        """
        Update progress for a single file.

        Args:
            filename: Name of file being processed
            vectors: Number of vectors created (if successful)
            status: One of 'processed', 'skipped', 'failed'
        """
        if status == "processed":
            self.processed += 1
            self.vectors_created += vectors
        elif status == "skipped":
            self.skipped += 1
        elif status == "failed":
            self.failed += 1

        if self.pbar:
            # Update tqdm with rich postfix information
            self.pbar.set_postfix({
                'OK': self.processed,
                'Skip': self.skipped,
                'Fail': self.failed,
                'Vectors': self.vectors_created
            })
            self.pbar.update(1)
        else:
            # Fallback: simple logging every 10 files
            total = self.processed + self.skipped + self.failed
            if total % 10 == 0 or total == self.total_files:
                logging.info(
                    "Progress: %d/%d files (%.1f%%) | Processed: %d | Skipped: %d | Failed: %d | Vectors: %d",
                    total,
                    self.total_files,
                    (total / self.total_files * 100) if self.total_files > 0 else 0,
                    self.processed,
                    self.skipped,
                    self.failed,
                    self.vectors_created
                )

    def write_message(self, message: str):
        """Write a message without disrupting progress bar."""
        if self.pbar:
            self.pbar.write(message)
        else:
            logging.info(message)

    def close(self):
        """Close the progress bar."""
        if self.pbar:
            self.pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# ============================================================================
# ENHANCED FILE LISTING WITH SECURITY
# ============================================================================


def list_local_files_secure(
    base_path: str,
    ext_whitelist: set[str],
    max_file_size_mb: float = 100
) -> list[dict[str, Any]]:
    """
    List files with security validation and size pre-filtering.

    Args:
        base_path: Base directory to scan
        ext_whitelist: Allowed file extensions
        max_file_size_mb: Skip files larger than this

    Returns:
        List of file metadata dicts

    Raises:
        FileNotFoundError: If base_path doesn't exist
        NotADirectoryError: If base_path is not a directory
    """
    base_path_obj = Path(base_path).resolve()

    if not base_path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    if not base_path_obj.is_dir():
        raise NotADirectoryError(f"Not a directory: {base_path}")

    files = []
    skipped_large = 0
    skipped_ext = 0

    for filepath in base_path_obj.rglob("*"):
        if not filepath.is_file():
            continue

        # Security: Skip symlinks to prevent following links outside base
        if filepath.is_symlink():
            logging.warning("Skipping symlink: %s", filepath)
            continue

        # Check extension whitelist
        file_ext = filepath.suffix[1:].lower() if filepath.suffix else ""
        if ext_whitelist and file_ext not in ext_whitelist:
            skipped_ext += 1
            continue

        # Pre-filter by size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if max_file_size_mb > 0 and size_mb > max_file_size_mb:
            logging.debug("Skipping large file: %s (%.2fMB)",
                          filepath.name, size_mb)
            skipped_large += 1
            continue

        # Create S3-like object dict
        try:
            relative_path = filepath.relative_to(base_path_obj)
            obj = {
                "Key": str(relative_path),
                "Size": filepath.stat().st_size,
                "LastModified": filepath.stat().st_mtime,
            }
            files.append(obj)
        except ValueError as e:
            logging.warning(
                "Could not get relative path for %s: %s", filepath, e)
            continue

    logging.info(
        "Found %d files (skipped %d by extension, %d by size)",
        len(files), skipped_ext, skipped_large
    )

    return files

# ---------------------------------------------------------------------------
# UPSERT VECTORS
# ---------------------------------------------------------------------------


def upsert_vectors(ctx, vectors: List[Dict[str, Any]]) -> None:
    """
    Upsert vectors to Pinecone index in batches.
    Expects each vector dict to have 'id', 'values', 'metadata', and 'namespace'.
    """
    if not vectors:
        return

    try:
        index = ctx.index
        grouped: Dict[Optional[str], List[Dict[str, Any]]] = {}

        # Group by namespace
        for v in vectors:
            ns = v.get("namespace")
            grouped.setdefault(ns, []).append(v)

        # Upsert per namespace
        from math import ceil

        MAX_REQUEST_VECTORS = 300  # ✅ Safe under 2 MB limit for 1536-dim embeddings

        for ns, vecs in grouped.items():
            clean_vecs = [{k: v for k, v in v.items() if k != "namespace"}
                          for v in vecs]

            # Split large batches into smaller chunks
            total = len(clean_vecs)
            chunks = ceil(total / MAX_REQUEST_VECTORS)
            for i in range(chunks):
                start = i * MAX_REQUEST_VECTORS
                end = start + MAX_REQUEST_VECTORS
                batch = clean_vecs[start:end]
                try:
                    index.upsert(vectors=batch, namespace=ns)
                    logging.info(
                        "Upserted %d/%d vectors into namespace '%s'",
                        len(batch), total, ns or "__default__"
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(
                        "Error during batch upsert (%d-%d): %s", start, end, e)

        ctx.stats.increment("total_vectors", len(vectors))

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error during upsert: %s", e, exc_info=True)
        ctx.stats.increment("files_failed")

# ---------------------------------------------------------------------------
# METADATA ENRICHMENT LOGIC
# ---------------------------------------------------------------------------


BUILDING_METADATA_FIELDS = [
    "Property code",
    "Property postcode",
    "Property campus",
    "UsrFRACondensedPropertyName",
    "Property names",
    "Property alternative names",
    "Property condition",
    "Property gross area (sq m)",
    "Property net area (sq m)",
]

MAX_ALIASES = 50
MAX_ALIAS_LEN = 120


def _merge_aliases(existing, incoming) -> list[str]:
    out = []
    seen = set()

    def add(x: str):
        x = (x or "").strip()
        if not x:
            return
        if len(x) > MAX_ALIAS_LEN:
            x = x[:MAX_ALIAS_LEN]
        k = x.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(x)

    for a in (existing or []):
        add(a)
    for a in (incoming or []):
        add(a)

    return out[:MAX_ALIASES]


def enrich_with_building_metadata(
    metadata: dict,
    canonical: str,
    ctx: "IngestContext",
    doc_type: str,
    chunk_idx: int,
    *,
    prefer_existing: bool = True,
) -> dict:
    """
    Centralised building metadata enrichment.

    - Always ensures building_aliases exists when canonical is present
    - Merges aliases (no overwrite)
    - Adds standard property fields from cache
    - Never overwrites existing metadata unless prefer_existing=False
    """
    if not canonical:
        return metadata

    canonical_key = canonical.strip()
    cached = ctx.cache.get_metadata(canonical_key)

    # Ensure aliases always exist and include canonical as a minimum useful identifier
    metadata["building_aliases"] = _merge_aliases(
        metadata.get("building_aliases"),
        [canonical_key],
    )

    if not cached:
        if chunk_idx == 0:
            logging.debug(
                "No cached building metadata for '%s' (%s)", canonical_key, doc_type)
        return metadata

    # Merge cached aliases
    if "building_aliases" in cached:
        metadata["building_aliases"] = _merge_aliases(
            metadata.get("building_aliases"),
            cached.get("building_aliases"),
        )
        if chunk_idx == 0:
            logging.info("✅ Added aliases for %s (%s): %d total",
                         canonical_key, doc_type, len(metadata["building_aliases"]))

    # Copy standard fields
    for field in BUILDING_METADATA_FIELDS:
        if field not in cached:
            continue
        if prefer_existing and field in metadata:
            continue
        metadata[field] = cached[field]

    return metadata
