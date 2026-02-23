#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingestion utility functions for Alfred Local.
This module provides various utility functions used during the ingestion process, including:
- Secure path validation to prevent path traversal attacks.
- Metadata validation with size estimation and truncation.
- Namespace routing validation based on document type.
- A progress tracker class that supports both tqdm and simple logging.
- A secure file listing function that pre-filters by size and validates file types.
- A helper to calculate safe Pinecone batch sizes based on embedding dimensions.
- A function to upsert vectors to Pinecone in batches with error handling and logging.
- Building metadata enrichment logic that merges cached data without overwriting existing values.
- A metrics exporter that outputs Prometheus text exposition format for monitoring.
These utilities are designed to support the main ingestion workflow while keeping the code modular and maintainable.
"""

import json
import logging
import uuid
import time
from typing import Any, Optional, TYPE_CHECKING
from math import ceil
from pathlib import Path
import mimetypes
from tqdm import tqdm
from config import (
    _route_namespace,
)
from alfred_exceptions import RoutingError, ValidationError, ExternalServiceError
from pinecone_utils import sanitise_metadata_for_pinecone

# ============================================================================
# INGEST CONTEXT
# ============================================================================
if TYPE_CHECKING:
    from .context import IngestContext


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger(__name__)

# ============================================================================
# PATH HANDLING
# ============================================================================


def validate_file_type(path: Path, allowed_types: set[str]) -> bool:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in allowed_types:
        raise ValidationError(f"File type not allowed: {mime_type}")
    return True


def validate_safe_path(
    base_path: str,
    key: str,
    *,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Validate that a file key is safe and within the base directory.
    Prevents path traversal attacks.

    Args:
        base_path: Base directory path
        key: Relative file path/key
        logger: Optional logger instance

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If the path is unsafe, outside base, missing, or not a file/dir
        ValueError: If the file extension is not allowed
    """
    allowed_extensions = {
        'pdf', 'txt', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'json', 'csv'
    }
    base = Path(base_path).resolve()
    if not base.exists():
        raise ValidationError(f"Base directory not found: {base_path}")
    if not base.is_dir():
        raise ValidationError(f"Base path is not a directory: {base_path}")

    target = (base / key).resolve()

    # Check if target is within base directory
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValidationError(
            f"Path traversal detected: '{key}' resolves outside base directory"
        ) from exc
    # Validate extension
    ext = target.suffix.lower().lstrip('.')
    if ext not in allowed_extensions:
        raise ValueError(f"File extension not allowed: {ext}")
    # Reject symlinks explicitly
    if target.is_symlink():
        raise ValidationError(f"Symlinks are not allowed: {key}")
    # Additional security checks
    if not target.exists():
        raise ValidationError(f"File not found: {key}")

    if not target.is_file():
        raise ValidationError(f"Not a regular file: {key}")

    # Check for suspicious patterns
    suspicious_patterns = ['..', '~', '$']
    log = _get_logger(logger)
    for pattern in suspicious_patterns:
        if pattern in str(key):
            log.warning(
                "Suspicious pattern '%s' found in path: %s",
                pattern, key
            )

    return target

# ============================================================================
# METADATA VALIDATOR
# ============================================================================


def truncate_by_tokens(text: str, encoder, max_tokens: int) -> str:
    if not text:
        return text
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def validate_with_truncation(
    ctx: "IngestContext",
    metadata: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> tuple[bool, Optional[str]]:
    """
    Centralised pattern:
    1) Estimate size (pre-truncation)
    2) Truncate text field
    3) Validate (rechecks size)
    """
    size_bytes, ok, size_error = MetadataValidator.estimate_size(
        metadata,
        max_metadata_size=ctx.config.max_metadata_size,
    )
    if not ok:
        return False, f"{size_error} ({size_bytes} bytes)"

    text_value = metadata.get("text", "")
    if isinstance(text_value, str):
        metadata["text"] = truncate_by_tokens(
            text_value,
            encoder=ctx.encoder,
            max_tokens=ctx.config.max_metadata_text_tokens,
        )

    return MetadataValidator.validate(
        metadata,
        max_metadata_size=ctx.config.max_metadata_size,
        logger=logger,
    )


class MetadataValidator:
    SCHEMA_VERSION = "1.0"
    REQUIRED_FIELDS = {"source_path", "key", "source", "text", "document_type"}

    @staticmethod
    def estimate_size(
        metadata: dict[str, Any],
        *,
        max_metadata_size: int,
    ) -> tuple[int, bool, Optional[str]]:
        """
        Estimate the size (bytes) of metadata BEFORE any truncation.

        Returns:
            (size_bytes, ok, error_message)
        """
        try:
            size = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            return 0, False, f"Serialisation error: {e}"

        if size > max_metadata_size:
            return size, False, "Metadata too large"
        return size, True, None

    @staticmethod
    def validate(
        metadata: dict[str, Any],
        *,
        max_metadata_size: int,
        logger: logging.Logger | None = None,
    ) -> tuple[bool, Optional[str]]:
        log = _get_logger(logger)
        metadata.setdefault("_schema_version",
                            MetadataValidator.SCHEMA_VERSION)
        missing = MetadataValidator.REQUIRED_FIELDS - metadata.keys()
        if missing:
            return False, f"Missing required fields: {missing}"

        nulls = [k for k, v in metadata.items() if v is None]
        if nulls:
            return False, f"Null values in {nulls}"
        # Check for deprecated fields
        deprecated = {"old_field_name", "legacy_prop"}
        found_deprecated = set(metadata.keys()) & deprecated
        if found_deprecated:
            log.warning(
                "Deprecated fields found: %s. Please update your extraction logic.",
                found_deprecated)
        try:
            size = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
            if size > max_metadata_size:
                return False, "Metadata too large"
        except Exception as e:
            return False, f"Serialisation error: {e}"

        return True, None

# ============================================================================
# VALIDATE NAMESPACE ROUTING
# ============================================================================


def validate_namespace_routing(
    doc_type: Optional[str],
    resolved_namespace: Optional[str],
) -> tuple[bool, str]:
    try:
        expected = _route_namespace(doc_type)
    except RoutingError as e:
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

    def __init__(
        self,
        total_files: int,
        use_tqdm: bool = True,
        progress_log_interval: int = 10,
        *,
        logger: logging.Logger | None = None,
    ):
        """
        InitialiSe progress tracker.

        Args:
            total_files: Total number of files to process
            use_tqdm: Whether to use tqdm progress bar (requires package)
        """
        self.total_files = total_files
        self.use_tqdm = use_tqdm
        self.progress_log_interval = progress_log_interval
        self.pbar = None

        # Statistics
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.vectors_created = 0

        self.logger = _get_logger(logger)

        if use_tqdm:
            try:
                self.pbar = tqdm(
                    total=total_files,
                    desc="Processing files",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            except ImportError:
                self.logger.warning(
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
            if total % self.progress_log_interval == 0 or total == self.total_files:
                self.logger.info(
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
            self.logger.info(message)

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
    max_file_size_mb: float = 100,
    *,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """
    List files with security validation and size pre-filtering.

    Args:
        base_path: Base directory to scan
        ext_whitelist: Allowed file extensions
        max_file_size_mb: Skip files larger than this
        logger: Optional logger instance

    Returns:
        List of file metadata dicts

    Raises:
        ValidationError: If base_path doesn't exist or is not a directory
    """
    base_path_obj = Path(base_path).resolve()

    if not base_path_obj.exists():
        raise ValidationError(f"Directory not found: {base_path}")

    if not base_path_obj.is_dir():
        raise ValidationError(f"Not a directory: {base_path}")

    files = []
    skipped_large = 0
    skipped_ext = 0

    log = _get_logger(logger)

    for filepath in base_path_obj.rglob("*"):
        if not filepath.is_file():
            continue

        # Security: Skip symlinks to prevent following links outside base
        if filepath.is_symlink():
            log.warning("Skipping symlink: %s", filepath)
            continue

        # Check extension whitelist
        file_ext = filepath.suffix[1:].lower() if filepath.suffix else ""
        if ext_whitelist and file_ext not in ext_whitelist:
            skipped_ext += 1
            continue

        # Pre-filter by size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if max_file_size_mb > 0 and size_mb > max_file_size_mb:
            log.debug(
                "Skipping large file: %s (%.2fMB)",
                filepath.name,
                size_mb,
            )
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
            log.warning(
                "Could not get relative path for %s: %s", filepath, e)
            continue

    log.info(
        "Found %d files (skipped %d by extension, %d by size)",
        len(files), skipped_ext, skipped_large
    )

    return files

# ============================================================================
# MAX BATCH HELPER
# ============================================================================


def calculate_max_batch_size(dimension: int, max_request_mb: float = 1.0) -> int:
    """
    Calculate a safe Pinecone upsert batch size based on embedding dimension.

    Pinecone request payloads should stay under ~2MB.
    Default uses a conservative 1.8MB safety margin.
    """
    bytes_per_vector = dimension * 4  # float32
    bytes_per_mb = 1024 * 1024
    max_bytes = max_request_mb * bytes_per_mb
    return max(1, int(max_bytes / bytes_per_vector))


# ============================================================================
# UPSERT VECTORS
# ============================================================================


def upsert_vectors(ctx, vectors: list[dict[str, Any]]) -> None:
    """
    Upsert vectors to Pinecone index in batches.
    Expects each vector dict to have 'id', 'values', 'metadata', and 'namespace'.
    """
    if not vectors:
        return

    try:
        store = ctx.vector_store
        grouped: dict[Optional[str], list[dict[str, Any]]] = {}

        # Group by namespace
        for v in vectors:
            ns = v.get("namespace")
            grouped.setdefault(ns, []).append(v)

        # Upsert per namespace

        MAX_REQUEST_VECTORS = calculate_max_batch_size(ctx.config.dimension)
        ctx.logger.debug(
            "Calculated MAX_REQUEST_VECTORS=%d for dimension=%d",
            MAX_REQUEST_VECTORS,
            ctx.config.dimension,
        )

        for ns, vecs in grouped.items():
            clean_vecs = []
            for v in vecs:
                clean_v = {k: v for k, v in v.items() if k in (
                    "id", "values", "metadata")}
                metadata = clean_v.get("metadata")
                if isinstance(metadata, dict):
                    clean_v["metadata"] = sanitise_metadata_for_pinecone(
                        metadata)
                elif metadata is None:
                    clean_v["metadata"] = {}
                clean_vecs.append(clean_v)

            # Split large batches into smaller chunks
            total = len(clean_vecs)
            chunks = ceil(total / MAX_REQUEST_VECTORS)
            for i in range(chunks):
                start = i * MAX_REQUEST_VECTORS
                end = start + MAX_REQUEST_VECTORS
                batch = clean_vecs[start:end]
                upsert_start = time.perf_counter()
                try:
                    store.upsert(vectors=batch, namespace=ns)
                    upsert_elapsed = time.perf_counter() - upsert_start
                    ctx.logger.info(
                        "Upserted %d/%d vectors into namespace '%s' in %.3fs",
                        len(batch), total, ns or "__default__", upsert_elapsed
                    )
                except Exception as e:  # pylint: disable=broad-except
                    namespace_label = ns or "__default__"
                    upsert_elapsed = time.perf_counter() - upsert_start
                    status = getattr(e, "status", None) or getattr(
                        e, "status_code", None
                    )
                    ctx.logger.warning(
                        "Upsert batch failed (ns=%s batch=%d-%d elapsed=%.3fs status=%s): %s",
                        namespace_label,
                        start,
                        end,
                        upsert_elapsed,
                        status,
                        e,
                    )
                    raise ExternalServiceError(
                        f"Upsert failed (namespace={namespace_label}, batch={start}-{end}, error={e})"
                    ) from e

        ctx.stats.increment("total_vectors", len(vectors))

    except Exception as e:  # pylint: disable=broad-except
        ctx.logger.error("Error during upsert: %s", e, exc_info=True)
        ctx.stats.increment("files_failed")
        raise


# ============================================================================
# METADATA ENRICHMENT LOGIC
# ============================================================================

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
            ctx.logger.debug(
                "No cached building metadata for '%s' (%s)", canonical_key, doc_type)
        return metadata

    # Merge cached aliases
    if "building_aliases" in cached:
        metadata["building_aliases"] = _merge_aliases(
            metadata.get("building_aliases"),
            cached.get("building_aliases"),
        )
        if chunk_idx == 0:
            ctx.logger.info(
                "✅ Added aliases for %s (%s): %d total",
                canonical_key,
                doc_type,
                len(metadata["building_aliases"]),
            )

    # Copy standard fields
    for field in BUILDING_METADATA_FIELDS:
        if field not in cached:
            continue
        if prefer_existing and field in metadata:
            continue
        metadata[field] = cached[field]

    return metadata


class MetricsExporter:
    """
    Export ingestion metrics in Prometheus text exposition format.

    This format works well with the node_exporter textfile collector, or can be
    scraped directly if you serve the file.
    """

    def export_prometheus(
        self,
        *,
        stats: dict[str, Any],
        output_path: str,
        duration_seconds: float,
        vectors_per_second: float,
        source_path: str,
        dry_run: bool,
        upsert_workers: int | None = None,
    ) -> None:
        # Prometheus label escaping
        def esc_label(value: object) -> str:
            s = str(value)
            s = s.replace("\\", "\\\\").replace(
                "\n", "\\n").replace('"', '\\"')
            return s

        run_id = str(stats.get("run_id") or uuid.uuid4().hex)
        source_id = Path(source_path).name or "unknown"
        labels = {
            "run_id": run_id,
            "source_id": source_id,
            "dry_run": str(dry_run).lower(),
            "index_name": str(stats.get("index_name") or "default"),
        }
        if upsert_workers is not None:
            labels["upsert_workers"] = str(upsert_workers)
        label_text = ",".join(
            f'{k}="{esc_label(v)}"' for k, v in labels.items()
        )
        run_info_labels = {
            **labels,
            "source_path": source_path,
        }
        run_info_text = ",".join(
            f'{k}="{esc_label(v)}"' for k, v in run_info_labels.items()
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Use atomic write so Prometheus never reads a partially written file
        tmp_path = out.with_suffix(out.suffix + ".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("# --- Run Info ---\n")
            f.write("# HELP askalfred_ingest_run_info Ingestion run metadata.\n")
            f.write("# TYPE askalfred_ingest_run_info gauge\n")
            f.write(
                f"askalfred_ingest_run_info{{{run_info_text}}} 1\n")

            # Counters (monotonic within a run)
            f.write("\n# --- File Counters ---\n")
            f.write(
                "# HELP askalfred_ingest_files_processed Total files processed.\n")
            f.write("# TYPE askalfred_ingest_files_processed counter\n")
            f.write(
                f"askalfred_ingest_files_processed{{{label_text}}} {int(stats.get('files_processed', 0))}\n")
            f.write("# HELP askalfred_ingest_files_skipped Total files skipped.\n")
            f.write("# TYPE askalfred_ingest_files_skipped counter\n")
            f.write(
                f"askalfred_ingest_files_skipped{{{label_text}}} {int(stats.get('files_skipped', 0))}\n")
            failed_files = stats.get("failed_files") or []
            failed_files_count = int(len(failed_files))
            f.write("# HELP askalfred_ingest_files_failed Total files failed.\n")
            f.write("# TYPE askalfred_ingest_files_failed counter\n")
            f.write(
                f"askalfred_ingest_files_failed{{{label_text}}} {failed_files_count}\n")

            f.write("\n# --- Vector Counters ---\n")
            f.write(
                "# HELP askalfred_ingest_total_vectors Total vectors upserted/created.\n")
            f.write("# TYPE askalfred_ingest_total_vectors counter\n")
            f.write(
                f"askalfred_ingest_total_vectors{{{label_text}}} {int(stats.get('total_vectors', 0))}\n")
            f.write(
                "# HELP askalfred_ingest_vectors_skipped Total vectors skipped.\n")
            f.write("# TYPE askalfred_ingest_vectors_skipped counter\n")
            f.write(
                f"askalfred_ingest_vectors_skipped{{{label_text}}} {int(stats.get('vectors_skipped', 0))}\n")

            # Gauges (point-in-time measurements)
            f.write("\n# --- Throughput ---\n")
            f.write(
                "# HELP askalfred_ingest_duration_seconds Ingestion duration in seconds.\n")
            f.write("# TYPE askalfred_ingest_duration_seconds gauge\n")
            f.write(
                f"askalfred_ingest_duration_seconds{{{label_text}}} {float(duration_seconds)}\n")
            f.write(
                "# HELP askalfred_ingest_vectors_per_second Average vectors per second.\n")
            f.write("# TYPE askalfred_ingest_vectors_per_second gauge\n")
            f.write(
                f"askalfred_ingest_vectors_per_second{{{label_text}}} {float(vectors_per_second)}\n")

            # Additional counters
            f.write("\n# --- Additional Counters ---\n")
            counters = [
                "embed_batches_total",
                "embed_texts_total",
                "embed_failed_total",
                "embed_retries_total",
                "embed_batch_reductions_total",
                "embed_rate_limit_total",
                "upsert_batches_total",
                "upsert_throttle_events_total",
                "upsert_batch_retries_total",
                "fra_lock_acquire_total",
                "fra_lock_acquire_attempts_total",
                "fra_lock_contended_total",
                "rollback_failures_total",
                "vectors_embedding_failed",
                "fra_embeddings_failed",
            ]
            for key in counters:
                if key in stats:
                    f.write(
                        f"# HELP askalfred_ingest_{key} {key.replace('_', ' ')}.\n")
                    f.write(
                        f"# TYPE askalfred_ingest_{key} counter\n")
                    f.write(
                        f"askalfred_ingest_{key}{{{label_text}}} {int(stats.get(key, 0))}\n")

            # Timing summaries (sum/count/max)
            f.write("\n# --- Timing Summaries ---\n")
            timing_bases = [
                "embed_batch_seconds",
                "upsert_batch_seconds",
                "upsert_backoff_sleep_seconds",
                "fra_lock_acquire_wait_seconds",
                "fra_supersession_global_lock_wait_seconds",
                "metadata_bytes",
                "metadata_bytes_ex_text",
                "metadata_bytes_post_validation",
                "upsert_batch_bytes",
                "upsert_batch_metadata_bytes_per_vector",
                "upsert_queue_put_wait_seconds",
            ]
            for base in timing_bases:
                count_key = f"{base}_count"
                sum_key = f"{base}_sum"
                max_key = f"{base}_max"
                if count_key in stats:
                    f.write(
                        f"# HELP askalfred_ingest_{count_key} {base} count.\n")
                    f.write(
                        f"# TYPE askalfred_ingest_{count_key} counter\n")
                    f.write(
                        f"askalfred_ingest_{count_key}{{{label_text}}} {int(stats.get(count_key, 0))}\n")
                if sum_key in stats:
                    f.write(
                        f"# HELP askalfred_ingest_{sum_key} {base} sum seconds.\n")
                    f.write(
                        f"# TYPE askalfred_ingest_{sum_key} counter\n")
                    f.write(
                        f"askalfred_ingest_{sum_key}{{{label_text}}} {float(stats.get(sum_key, 0.0))}\n")
                if max_key in stats:
                    f.write(
                        f"# HELP askalfred_ingest_{max_key} {base} max seconds.\n")
                    f.write(
                        f"# TYPE askalfred_ingest_{max_key} gauge\n")
                    f.write(
                        f"askalfred_ingest_{max_key}{{{label_text}}} {float(stats.get(max_key, 0.0))}\n")

            # Optional: expose failed file count as gauge (easy alerting)
            f.write("\n# --- Run Health ---\n")
            f.write(
                "# HELP askalfred_ingest_failed_files_count Number of failed files in this run.\n")
            f.write("# TYPE askalfred_ingest_failed_files_count gauge\n")
            f.write(
                f"askalfred_ingest_failed_files_count{{{label_text}}} {failed_files_count}\n")

        tmp_path.replace(out)


class DryRunIndex:
    """Mock Pinecone index for dry-run mode."""

    def __init__(self, logger):
        self.logger = logger

    def upsert(self, vectors, namespace=None):
        self.logger.info(
            f"[DRY-RUN] Would upsert {len(vectors)} vectors to {namespace}")
        return {"upserted_count": len(vectors)}

    def query(self, **kwargs):
        self.logger.info(f"[DRY-RUN] Would query with {kwargs}")
        return {"matches": []}

    def list(self, **kwargs):
        self.logger.info(f"[DRY-RUN] Would list with {kwargs}")
        return []

    def fetch(self, **kwargs):
        self.logger.info(f"[DRY-RUN] Would fetch with {kwargs}")

        class _Result:
            vectors = {}
        return _Result()

    def update(self, **kwargs):
        self.logger.info(f"[DRY-RUN] Would update with {kwargs}")
        return {"updated_count": 0}
