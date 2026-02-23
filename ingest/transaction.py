"""
Transaction management for Alfred Local ingestion.
This module provides classes and functions to handle transactional operations during the ingestion process,
including thread-safe statistics tracking, retry mechanisms, and FRA-specific processing.
"""
import logging
import time
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Any, Optional, TYPE_CHECKING
from interfaces import FileRecord, MetricsReader
from config import (
    FRA_RISK_ITEMS_NAMESPACE,
    INGEST_VERIFY_FETCH_BATCH_SIZE,
    INGEST_VERIFY_BACKOFF_BASE,
    INGEST_VERIFY_BACKOFF_CAP,
    FRA_LOCK_TIMEOUT_SECONDS,
    get_display_namespace,
    DocumentTypes,
    INGEST_VECTOR_BUFFER_MAX_SIZE,
)
from config.constant import INGEST_METADATA_CACHE_SIZE
from fra import (
    restore_superseded_items,
    mark_superseded_risk_items,
    deduplicate_risk_items,
    _fra_partition_key,
    FRAActionPlanParser,
    parse_action_plan_in_process,
    sanitise_risk_item_for_metadata,
    FRATriageComputer,
    EnrichedRiskItem,
    FraVectorExtractResult,
)
from alfred_exceptions import (
    ExternalServiceError,
    IngestError,
    ModelNotInitialisedError,
    ParseError,
    RoutingError,
    ValidationError,
    RollbackError,
)
from pinecone_utils import NULL_SENTINEL
from .document_content import (
    backoff_sleep,
    embed_texts_batch,
    ext,
)
from .utils import (
    validate_with_truncation,
    upsert_vectors,
)
from .helpers import _extract_fra_layout_text
if TYPE_CHECKING:
    from .context import IngestContext

# ============================================================================
# THREAD-SAFE STATS
# ============================================================================


class ThreadSafeStats(MetricsReader):
    def __init__(self):
        self._lock = Lock()
        self._stats = {
            "run_id": uuid.uuid4().hex,
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

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return self._stats.copy()

    def observe_timing(self, key: str, value: float) -> None:
        with self._lock:
            count_key = f"{key}_count"
            sum_key = f"{key}_sum"
            max_key = f"{key}_max"
            self._stats[count_key] = self._stats.get(count_key, 0) + 1
            self._stats[sum_key] = self._stats.get(sum_key, 0.0) + float(value)
            current_max = self._stats.get(max_key, 0.0)
            if float(value) > float(current_max):
                self._stats[max_key] = float(value)

    def observe_histogram(self, key: str, value: float) -> None:
        """Record a value for a histogram metric (e.g. metadata size)."""
        with self._lock:
            count_key = f"{key}_count"
            sum_key = f"{key}_sum"
            max_key = f"{key}_max"
            self._stats[count_key] = self._stats.get(count_key, 0) + 1
            self._stats[sum_key] = self._stats.get(sum_key, 0.0) + float(value)
            current_max = self._stats.get(max_key, 0.0)
            if float(value) > float(current_max):
                self._stats[max_key] = float(value)


# ============================================================================
# THREAD-SAFE CACHE
# ============================================================================

class ThreadSafeCache:
    def __init__(self, metadata_cache_size: int = INGEST_METADATA_CACHE_SIZE):
        self._lock = RLock()
        self._name_cache: dict[str, str] = {}
        self._alias_cache: dict[str, str] = {}
        self._metadata_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._metadata_cache_size = metadata_cache_size
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()

    def update_from_csv(
        self,
        name_to_canonical: dict[str, str],
        alias_to_canonical: dict[str, str],
        metadata_cache: dict[str, dict[str, Any]],
    ) -> None:
        with self._lock:
            self._name_cache.update(name_to_canonical)
            self._alias_cache.update(alias_to_canonical)
            for k, v in metadata_cache.items():
                self._metadata_cache[k] = v.copy()
                self._metadata_cache.move_to_end(k)
                if len(self._metadata_cache) > self._metadata_cache_size:
                    self._metadata_cache.popitem(last=False)

    def get_name_mapping(self) -> dict[str, str]:
        with self._lock:
            return self._name_cache.copy()

    def get_alias_mapping(self) -> dict[str, str]:
        with self._lock:
            return self._alias_cache.copy()

    def get_metadata(self, building_name: str) -> Optional[dict[str, Any]]:
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
            if metadata is not None:
                # LRU: mark most recently used
                self._metadata_cache.move_to_end(building_name)
            # Return a copy to prevent external modifications
            return metadata.copy() if metadata else None

    def set_metadata(self, building_name: str, metadata: dict[str, Any]) -> None:
        """
        Cache metadata for a building.

        Args:
            building_name: The canonical building name
            metadata: Metadata dictionary to cache
        """
        with self._lock:
            self._metadata_cache[building_name] = metadata.copy()
            self._metadata_cache.move_to_end(building_name)
            if len(self._metadata_cache) > self._metadata_cache_size:
                self._metadata_cache.popitem(last=False)

    def has_metadata(self, building_name: str) -> bool:
        """Check if metadata exists for a building."""
        with self._lock:
            return building_name in self._metadata_cache

    def invalidate_building(self, building: str) -> None:
        with self._lock:
            self._name_cache.pop(building, None)
            self._alias_cache.pop(building, None)
            self._metadata_cache.pop(building, None)
            self._embedding_cache.pop(building, None)

    def clear_all(self) -> None:
        with self._lock:
            self._name_cache.clear()
            self._alias_cache.clear()
            self._metadata_cache.clear()
            self._embedding_cache.clear()

    def get_embedding(self, building_name: str) -> list[float] | None:
        with self._lock:
            embedding = self._embedding_cache.get(building_name)
            if embedding is not None:
                self._embedding_cache.move_to_end(building_name)
                return list(embedding)
            return None

    def set_embedding(self, building_name: str, embedding: list[float]) -> None:
        with self._lock:
            self._embedding_cache[building_name] = list(embedding)
            self._embedding_cache.move_to_end(building_name)
            if len(self._embedding_cache) > self._metadata_cache_size:
                self._embedding_cache.popitem(last=False)


# ============================================================================
# THREAD-SAFE VECTOR BUFFER
# ============================================================================


class ThreadSafeVectorBuffer:
    """Thread-safe buffer for pending vectors."""

    def __init__(self, max_size: int = INGEST_VECTOR_BUFFER_MAX_SIZE):
        self._lock = RLock()
        self._buffer: list[dict[str, Any]] = []
        self.max_size = max_size

    def add(self, vector: dict[str, Any], auto_flush_callback=None) -> bool:
        """
        Add vector with optional auto-flush.

        Returns:
            True if added, False if buffer full (and auto_flush not provided)
        """
        with self._lock:
            if len(self._buffer) >= self.max_size:
                if auto_flush_callback:
                    # Auto-flush when full
                    to_flush = self._buffer[:]
                    self._buffer.clear()
                    auto_flush_callback(to_flush)
                    self._buffer.append(vector)
                    return True
                else:
                    return False  # Let caller decide how to handle
            self._buffer.append(vector)
            return True

    def get_and_clear(self) -> list[dict[str, Any]]:
        """Retrieve and clear all buffered vectors (used for upserts)."""
        with self._lock:
            data = self._buffer[:]
            self._buffer.clear()
            return data

    def extend(self, vectors: list[dict[str, Any]]) -> None:
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


class FraSupersessionTxnLog:
    """Lightweight in-memory transaction log for supersession rollback tracking."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._tx_superseded: dict[str, list[str]] = {}
        self.logger = logger or logging.getLogger(__name__)

    def begin(self, buildings: list[str], request_count: int) -> str:
        tx_id = uuid.uuid4().hex
        self._tx_superseded[tx_id] = []
        self.logger.info(
            "FRA txn begin: %s (buildings=%d, requests=%d)",
            tx_id,
            len(buildings),
            request_count,
        )
        return tx_id

    def record_superseded(self, tx_id: str, building: str, ids: list[str]) -> None:
        if tx_id not in self._tx_superseded:
            return
        self._tx_superseded[tx_id].extend(ids)
        self.logger.info(
            "FRA txn record: %s (building=%s, superseded=%d)",
            tx_id,
            building,
            len(ids),
        )

    def get_superseded(self, tx_id: str) -> list[str]:
        return list(self._tx_superseded.get(tx_id, []))

    def commit(self, tx_id: str) -> None:
        if tx_id in self._tx_superseded:
            self.logger.info(
                "FRA txn commit: %s (superseded=%d)",
                tx_id,
                len(self._tx_superseded[tx_id]),
            )
            self._tx_superseded.pop(tx_id, None)

    def rollback(self, tx_id: str, ctx: "IngestContext", reason: str) -> int:
        """Rollback superseded items and return count restored."""
        if tx_id not in self._tx_superseded:
            return 0

        superseded_ids = self._tx_superseded[tx_id]
        self.logger.warning(
            "FRA txn rollback: %s (restoring %d items, reason=%s)",
            tx_id, len(superseded_ids), reason
        )
        # Call the restore function from fra_integration
        restored = restore_superseded_items(ctx, superseded_ids)
        self._tx_superseded.pop(tx_id, None)  # Clear the log after rollback
        if superseded_ids and restored < len(superseded_ids):
            raise RollbackError(
                f"Critical rollback failure: restored {restored}/{len(superseded_ids)} superseded items "
                f"(tx_id={tx_id}, reason={reason})"
            )
        return restored


# ---------------------------------------------------------------------------
# 6. FraTransaction  (explicit prepare / execute / verify / rollback)
# ---------------------------------------------------------------------------

class FraTransaction:
    """
    Encapsulates the four phases of an FRA atomic upsert:

      prepare()  — collect supersession requests, acquire locks
      execute()  — mark superseded items, upsert vectors
      verify()   — confirm vectors are present in the index
      rollback() — restore superseded items on failure

    Locks are held between prepare() and either verify() or rollback().
    Use as a context manager for the lock scope, or call phases explicitly.
    """

    def __init__(self, ctx: "IngestContext", vectors: list[dict[str, Any]]) -> None:
        self._ctx = ctx
        self._vectors = vectors
        self._supersede_requests: list[tuple[str, str]] = []
        self._superseded_ids: list[str] = []
        self._txn_log: FraSupersessionTxnLog | None = None
        self._tx_id: str | None = None
        self._lock_ctx = None
        self._lock_lost_event: threading.Event = (
            ctx.upsert_stop_event or threading.Event()
        )

    # -- phase helpers --

    def prepare(self) -> bool:
        """
        Collect supersession requests and check the registry.
        Returns True if a transactional supersession is required,
        False if a simple upsert is sufficient.
        """
        requests = _collect_fra_supersede_requests(self._vectors)
        self._supersede_requests = _filter_supersede_requests_with_registry(
            self._ctx, requests
        )
        return bool(self._supersede_requests)

    def acquire_locks(self):
        """
        Acquire Redis locks for all buildings involved.
        Returns the lock context manager (caller must enter it).
        """
        buildings = sorted({b for b, _ in self._supersede_requests})
        self._txn_log = FraSupersessionTxnLog(logger=self._ctx.logger)
        self._tx_id = self._txn_log.begin(
            buildings, len(self._supersede_requests))
        return _acquire_fra_locks(
            self._ctx,
            buildings,
            timeout_seconds=FRA_LOCK_TIMEOUT_SECONDS,
            lock_lost_event=self._lock_lost_event,
        )

    def execute(self) -> None:
        """Mark superseded items and upsert vectors. Must hold locks."""
        self._raise_if_lock_lost()
        self._ctx.logger.info(
            "FRA supersession batch: %d building/date pairs",
            len(self._supersede_requests),
        )
        for building, assessment_date in self._supersede_requests:
            self._raise_if_lock_lost()
            self._ctx.logger.info(
                "Superseding FRA items for %s (new assessment: %s)",
                building,
                assessment_date,
            )

        for building, assessment_date in self._supersede_requests:
            self._raise_if_lock_lost()
            superseded = mark_superseded_risk_items(
                ctx=self._ctx,
                building=building,
                new_assessment_date=assessment_date,
            )
            self._superseded_ids.extend(superseded)
            if self._txn_log is not None:
                self._txn_log.record_superseded(
                    self._tx_id, building, superseded)  # type: ignore[arg-type]

        self._raise_if_lock_lost()
        upsert_vectors(self._ctx, self._vectors)
        _mark_batch_state(self._ctx, self._vectors, status="upserted")

    def verify(self) -> list[str]:
        """
        Confirm all FRA vectors are present in the index.
        Returns a list of missing IDs (empty on success).
        Commits the txn log on success.
        """
        self._raise_if_lock_lost()
        missing_ids = _verify_fra_vectors_present(self._ctx, self._vectors)
        if not missing_ids:
            if self._txn_log is not None:
                self._txn_log.commit(self._tx_id)  # type: ignore[arg-type]
        return missing_ids

    def rollback(self, reason: str) -> None:
        """Restore superseded items and roll back the txn log."""
        if self._txn_log and self._tx_id:
            self._txn_log.rollback(self._tx_id, self._ctx, reason=reason)

    # -- private --

    def _raise_if_lock_lost(self) -> None:
        if self._lock_lost_event.is_set():
            raise ExternalServiceError(
                "Redis lock lost during FRA supersession")


# ---------------------------------------------------------------------------
# upsert_vectors_atomic  (refactored — delegates to FraTransaction)
# ---------------------------------------------------------------------------

def _emit_verification_failure_event(
    ctx: "IngestContext",
    missing_ids: list[str],
) -> None:
    try:
        ctx.event_sink.emit_event(
            {
                "event_type": "fra_verification_failed",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "missing_count": len(missing_ids),
                "missing_ids": missing_ids[:50],
                "namespace": FRA_RISK_ITEMS_NAMESPACE,
            }
        )
    except Exception as alert_error:  # pylint: disable=broad-except
        ctx.logger.warning(
            "Verification failure alert emission failed: %s", alert_error)


def _handle_verification_failure(
    ctx: "IngestContext",
    vectors: list[dict[str, Any]],
    missing_ids: list[str],
) -> None:
    _emit_verification_failure_event(ctx, missing_ids)
    error_tag = f"verification_failed_missing_{len(missing_ids)}"
    _mark_batch_state(ctx, vectors, status="failed", error=error_tag)
    try:
        _record_ingested_files(ctx, vectors, status="failed", error=error_tag)
    except Exception as record_error:  # pylint: disable=broad-except
        ctx.logger.warning("FileRegistry update failed: %s", record_error)
    raise RuntimeError(
        f"FRA upsert verification failed; missing {len(missing_ids)} vectors"
    )


def upsert_vectors_atomic(ctx: "IngestContext", vectors: list[dict[str, Any]]) -> None:
    """
    Two-phase commit for FRA risk items:
      1. Prepare — collect supersession requests
      2. Execute — mark superseded items + upsert
      3. Verify  — confirm vectors present
      4. Rollback on any failure
    """
    if not vectors:
        return

    txn = FraTransaction(ctx, vectors)
    needs_supersession = txn.prepare()

    if not needs_supersession:
        # Simple path: no supersession needed
        upsert_vectors(ctx, vectors)
        _mark_batch_state(ctx, vectors, status="upserted")
        missing_ids = _verify_fra_vectors_present(ctx, vectors)
        if missing_ids:
            _handle_verification_failure(ctx, vectors, missing_ids)
        _mark_batch_state(ctx, vectors, status="verified")
        try:
            _record_ingested_files(ctx, vectors, status="success")
        except Exception as error:  # pylint: disable=broad-except
            ctx.logger.warning("FileRegistry update failed: %s", error)
        return

    # Transactional path: acquire locks and run all phases
    with txn.acquire_locks():
        try:
            txn.execute()
            missing_ids = txn.verify()
            if missing_ids:
                _handle_verification_failure(ctx, vectors, missing_ids)

            _mark_batch_state(ctx, vectors, status="verified")
            try:
                _record_ingested_files(ctx, vectors, status="success")
            except Exception as error:  # pylint: disable=broad-except
                ctx.logger.warning("FileRegistry update failed: %s", error)

        except (ExternalServiceError, IngestError, ValidationError, RoutingError,
                ParseError, ModelNotInitialisedError) as error:
            txn.rollback(str(error))
            _mark_batch_state(ctx, vectors, status="failed", error=str(error))
            try:
                _record_ingested_files(
                    ctx, vectors, status="failed", error=str(error))
            except Exception as record_error:  # pylint: disable=broad-except
                ctx.logger.warning(
                    "FileRegistry update failed: %s", record_error)
            raise


# ---------------------------------------------------------------------------
# FRA supersession helpers
# ---------------------------------------------------------------------------

def _acquire_fra_locks(
    ctx: "IngestContext",
    buildings: list[str],
    timeout_seconds: float,
    lock_lost_event: threading.Event | None = None,
):  # pylint: disable=unused-argument
    """Acquire Redis locks for buildings (assumed de-duplicated)."""
    if getattr(ctx.config, "fra_supersession_single_threaded", False):
        manager = ctx.redis_locks
        building_locks = sorted(buildings)

        class _Ctx:
            def __init__(self):
                self._global_ctx = None
                self._building_ctx = None

            def __enter__(self):
                # Global lock to serialize supersession across workers.
                start = time.monotonic()
                self._global_ctx = manager.lock(
                    "__global__",
                    ttl_ms=int(timeout_seconds * 1000),
                    auto_renew=True,
                    lock_lost_event=lock_lost_event,
                )
                self._global_ctx.__enter__()
                elapsed = time.monotonic() - start
                try:
                    ctx.stats.observe_timing(
                        "fra_supersession_global_lock_wait_seconds", elapsed)
                except Exception:
                    pass
                ctx.logger.info(
                    "FRA supersession global lock acquired in %.3fs", elapsed
                )
                self._building_ctx = manager.lock_many(
                    building_locks,
                    auto_renew=True,
                    lock_lost_event=lock_lost_event,
                )
                return self._building_ctx.__enter__()

            def __exit__(self, exc_type, exc, tb) -> None:
                if self._building_ctx is not None:
                    self._building_ctx.__exit__(exc_type, exc, tb)
                if self._global_ctx is not None:
                    self._global_ctx.__exit__(exc_type, exc, tb)
                    ctx.logger.info(
                        "FRA supersession global lock released"
                    )

        return _Ctx()

    return ctx.redis_locks.lock_many(
        sorted(buildings),
        auto_renew=True,
        lock_lost_event=lock_lost_event,
    )


def _collect_fra_supersede_requests(
    vectors: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    requests: dict[tuple[str, str], bool] = {}
    for vector in vectors:
        if vector.get("namespace") != FRA_RISK_ITEMS_NAMESPACE:
            continue
        metadata = vector.get("metadata") or {}
        building = metadata.get("canonical_building_name")
        assessment_date = metadata.get("fra_assessment_date")
        if assessment_date == NULL_SENTINEL:
            assessment_date = None
        if building and assessment_date:
            requests[(building, assessment_date)] = True
    return list(requests.keys())


def _filter_supersede_requests_with_registry(
    ctx: "IngestContext",
    requests: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if not requests:
        return []
    filtered: list[tuple[str, str]] = []
    for building, assessment_date in requests:
        registry_id = f"fra_supersede:{building}:{assessment_date}"
        try:
            existing = ctx.job_registry.get(registry_id)
        except Exception as error:  # pylint: disable=broad-except
            ctx.logger.warning(
                "JobRegistry lookup failed for %s: %s", registry_id, error)
            existing = None
        if existing and existing.status == "success":
            ctx.logger.info(
                "Supersession already recorded for %s (assessment: %s); skipping",
                building,
                assessment_date,
            )
            continue
        filtered.append((building, assessment_date))
    return filtered


def _verify_fra_vectors_present(
    ctx: "IngestContext",
    vectors: list[dict[str, Any]],
    attempts: int = 1,
) -> list[str]:
    fra_ids = [
        vector.get("id")
        for vector in vectors
        if vector.get("namespace") == FRA_RISK_ITEMS_NAMESPACE
    ]
    fra_ids = [vector_id for vector_id in fra_ids if vector_id]
    if not fra_ids:
        return []

    batch_size = INGEST_VERIFY_FETCH_BATCH_SIZE
    missing_ids = set(fra_ids)

    for attempt in range(attempts):
        still_missing = set()
        for i in range(0, len(fra_ids), batch_size):
            batch = fra_ids[i:i + batch_size]
            try:
                response = ctx.vector_store.fetch(
                    ids=batch, namespace=FRA_RISK_ITEMS_NAMESPACE)
            except ExternalServiceError as error:
                ctx.logger.error(
                    "FRA verification fetch failed (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    error,
                )
                still_missing.update(batch)
                continue

            if not response or not getattr(response, "vectors", None):
                still_missing.update(batch)
                continue

            found_ids = set(response.vectors.keys())
            still_missing.update(set(batch) - found_ids)

        if not still_missing:
            return []

        missing_ids = still_missing
        if attempt + 1 < attempts:
            backoff_sleep(
                attempt + 1,
                base=INGEST_VERIFY_BACKOFF_BASE,
                cap=INGEST_VERIFY_BACKOFF_CAP,
            )

    return sorted(missing_ids)


# ---------------------------------------------------------------------------
# Batch state / file registry helpers (unchanged)
# ---------------------------------------------------------------------------

def _record_ingested_files(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    status: str,
    error: str | None = None,
) -> None:
    if getattr(ctx.config, "dry_run", False):
        return

    ingested_at_iso = datetime.now(timezone.utc).isoformat() + "Z"
    records: dict[str, dict[str, Any]] = {}
    namespaces: dict[str, set[str]] = {}
    tokens: dict[str, str | None] = {}

    for vector in batch:
        vector_id = vector.get("id")
        if not vector_id:
            continue
        file_id = str(vector_id).split(":", 1)[0]
        if not file_id:
            continue
        metadata = vector.get("metadata") or {}
        ns = get_display_namespace(vector.get("namespace"))
        namespaces.setdefault(file_id, set()).add(ns)
        if file_id not in tokens:
            tokens[file_id] = vector.get("_processing_token")

        if file_id not in records:
            records[file_id] = {
                "file_id": file_id,
                "source_path": metadata.get("source_path", ""),
                "source_key": metadata.get("key") or metadata.get("source") or "",
                "content_hash": metadata.get("content_hash"),
                "ingested_at_iso": ingested_at_iso,
                "status": status,
                "error": error,
            }

    for file_id, payload in records.items():
        ns_tuple = tuple(sorted(namespaces.get(file_id, set())))
        ctx.file_registry.upsert_with_token(
            FileRecord(
                file_id=payload["file_id"],
                source_path=payload["source_path"],
                source_key=payload["source_key"],
                content_hash=payload["content_hash"],
                ingested_at_iso=payload["ingested_at_iso"],
                namespaces=ns_tuple,
                status=payload["status"],
                error=payload["error"],
            ),
            processing_token=tokens.get(file_id),
        )


def _mark_batch_state(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    status: str,
    error: str | None = None,
) -> None:
    for vector in batch:
        vector_id = vector.get("id")
        if not vector_id:
            continue
        file_id = str(vector_id).split(":", 1)[0]
        if not file_id:
            continue
        metadata = vector.get("metadata") or {}
        ns = get_display_namespace(vector.get("namespace"))
        processing_token = vector.get("_processing_token")
        try:
            ctx.file_registry.mark_state(
                file_id=file_id,
                processing_token=processing_token,
                status=status,
                error=error,
                source_path=metadata.get("source_path", ""),
                source_key=metadata.get("key") or metadata.get("source") or "",
                content_hash=metadata.get("content_hash"),
                namespaces=(ns,) if ns else (),
            )
        except Exception as err:  # pylint: disable=broad-except
            ctx.logger.warning("FileRegistry state update failed: %s", err)


def _mark_batch_failed(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    reason: str,
) -> None:
    if not batch:
        return
    _mark_batch_state(ctx, batch, status="failed", error=reason)
    try:
        _record_ingested_files(ctx, batch, status="failed", error=reason)
    except Exception as error:  # pylint: disable=broad-except
        ctx.logger.warning("FileRegistry update failed: %s", error)


# ---------------------------------------------------------------------------
# FRA risk-item extraction
# ---------------------------------------------------------------------------

def _create_risk_item_summary(item: EnrichedRiskItem) -> str:
    """Create a searchable summary text for an FRA risk item."""

    def _as_str(value: object, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    expected_date = _as_str(item.get("expected_completion_date"), "Not set")
    actual_date = _as_str(item.get("actual_completion_date"), "Not completed")
    issue_number = _as_str(item.get("issue_number"), "Unknown")
    risk_level_text = _as_str(item.get("risk_level_text"), "Unknown")
    risk_level = _as_str(item.get("risk_level"), "?")
    building_name = _as_str(
        item.get("canonical_building_name"), "Unknown building")
    risk_category = _as_str(item.get("risk_category"),
                            "other").replace("_", " ").title()
    completion_status = _as_str(item.get("completion_status"), "open").upper()
    issue_description = _as_str(
        item.get("issue_description"),
        "No issue description provided.",
    )
    proposed_solution = _as_str(
        item.get("proposed_solution"),
        "No proposed solution provided.",
    )
    person_responsible = _as_str(
        item.get("person_responsible"), "Not assigned")
    job_reference = _as_str(item.get("job_reference"), "No job reference")

    summary = f"""
Risk Item #{issue_number} - {risk_level_text} Risk (Level {risk_level}/5)
Building: {building_name}
Category: {risk_category}
Status: {completion_status}

Issue Description:
{issue_description}

Proposed Solution:
{proposed_solution}

Responsibility:
Assigned to: {person_responsible}
Job Reference: {job_reference}

Timeline:
Expected Completion: {expected_date}
Actual Completion: {actual_date}
"""

    if item.get("requires_immediate_action"):
        summary += "\nREQUIRES IMMEDIATE ACTION"
    elif item.get("requires_attention"):
        summary += "\nREQUIRES ATTENTION"

    if item.get("flag_overdue"):
        days = _as_str(item.get("days_overdue"), "0")
        summary += f"\nOVERDUE by {days} days"

    if item.get("flag_high_risk_no_job"):
        summary += "\nHIGH RISK - NO JOB REFERENCE"

    figure_refs = item.get("figure_references")
    if isinstance(figure_refs, list) and figure_refs:
        figures = ", ".join(str(fig) for fig in figure_refs)
        summary += f"\n\nEvidence: See Figure(s) {figures}"

    return summary.strip()


def extract_fra_risk_items_integration(
    ctx: "IngestContext",
    *,
    base_path: str,
    key: str,
    text_sample: str,
    building: str,
    content_hash: str | None,
    file_id: str,
    processing_token: str,
    start_time: float,
    vectors_to_upsert: list[dict[str, Any]],
    parse_pool: ProcessPoolExecutor | None = None,
) -> FraVectorExtractResult:
    """
    Extract and embed FRA risk items from a candidate FRA file.
    """
    page_texts = None
    parse_text = text_sample
    raw_text_sample = text_sample
    parse_verbose = ctx.config.log_level == "DEBUG"
    triage_computer = FRATriageComputer(verbose=parse_verbose)

    used_layout_text = False
    if ext(key) == "pdf":
        layout_text = _extract_fra_layout_text(
            ctx,
            base_path=base_path,
            key=key,
        )
        if layout_text:
            parse_text = layout_text
            used_layout_text = True

    if parse_pool:
        parse_start = time.perf_counter()
        future = parse_pool.submit(
            parse_action_plan_in_process,
            parse_text,
            key,
            building,
            parse_verbose,
        )
        risk_items, confidence = future.result()
        parse_elapsed = time.perf_counter() - parse_start
        ctx.logger.debug(
            "FRA parse (process) %s: %.3fs",
            key,
            parse_elapsed,
        )
    else:
        fra_parser = FRAActionPlanParser(verbose=parse_verbose)
        parse_start = time.perf_counter()
        risk_items, confidence = fra_parser.extract_risk_items(
            item_text=parse_text,
            item_key=key,
            canonical_building=building,
            page_texts=page_texts,
        )
        parse_elapsed = time.perf_counter() - parse_start
        ctx.logger.debug(
            "FRA parse (thread) %s: %.3fs",
            key,
            parse_elapsed,
        )

    parsing_confidence = getattr(confidence, "overall", 0.0)
    parsing_warnings = list(getattr(confidence, "warnings", []) or [])
    parsing_field_scores = dict(getattr(confidence, "field_scores", {}) or {})

    def _field_score_below(
        scores: dict[str, float],
        field_name: str,
        threshold: float,
    ) -> bool:
        value = scores.get(field_name)
        return value is not None and value < threshold

    if used_layout_text and raw_text_sample:
        low_confidence = parsing_confidence < 0.35
        low_fields = (
            _field_score_below(parsing_field_scores, "issue_description", 0.5)
            or _field_score_below(parsing_field_scores, "proposed_solution", 0.5)
        )
        if low_confidence or low_fields:
            ctx.logger.warning(
                "Low-quality FRA layout parse for %s (confidence %.2f); retrying text parse",
                key,
                parsing_confidence,
            )
            if parse_pool:
                retry_future = parse_pool.submit(
                    parse_action_plan_in_process,
                    raw_text_sample,
                    key,
                    building,
                    parse_verbose,
                )
                retry_items, retry_confidence = retry_future.result()
            else:
                retry_parser = FRAActionPlanParser(verbose=parse_verbose)
                retry_items, retry_confidence = retry_parser.extract_risk_items(
                    item_text=raw_text_sample,
                    item_key=key,
                    canonical_building=building,
                    page_texts=page_texts,
                )
            retry_overall = getattr(retry_confidence, "overall", 0.0)
            if retry_overall >= parsing_confidence:
                risk_items = retry_items
                confidence = retry_confidence
                parsing_confidence = retry_overall
                parsing_warnings = list(
                    getattr(retry_confidence, "warnings", []) or []
                )
                parsing_field_scores = dict(
                    getattr(retry_confidence, "field_scores", {}) or {}
                )

    missing_action_plan = (
        not risk_items and
        "No action plan section found" in parsing_warnings
    )

    if missing_action_plan:
        ctx.stats.increment("fra_action_plan_missing")
        ctx.logger.warning("FRA action plan missing: %s", key)

    if not risk_items:
        ctx.logger.info("No risk items extracted from %s", key)
        return {
            "added": 0,
            "parsing_confidence": parsing_confidence,
            "parsing_warnings": parsing_warnings,
            "parsing_field_scores": parsing_field_scores,
            "missing_action_plan": missing_action_plan,
        }

    ctx.logger.info(
        "Extracted %d risk items from %s (confidence: %.2f)",
        len(risk_items),
        key,
        parsing_confidence,
    )

    enriched_items: list[EnrichedRiskItem] = [
        triage_computer.compute_flags(item) for item in risk_items
    ]
    enriched_items = deduplicate_risk_items(ctx, enriched_items)
    if not enriched_items:
        ctx.logger.info("All FRA risk items already exist for %s", key)
        return {
            "added": 0,
            "parsing_confidence": parsing_confidence,
            "parsing_warnings": parsing_warnings,
            "parsing_field_scores": parsing_field_scores,
            "missing_action_plan": missing_action_plan,
        }

    if getattr(ctx.config, "dry_run", False):
        ctx.logger.info(
            "Dry-run: skipping FRA risk item embeddings for %s", key)
        return {
            "added": 0,
            "parsing_confidence": parsing_confidence,
            "parsing_warnings": parsing_warnings,
            "parsing_field_scores": parsing_field_scores,
            "missing_action_plan": missing_action_plan,
        }

    summaries: list[str] = []
    for item in enriched_items:
        summaries.append(_create_risk_item_summary(item))

    max_seconds = getattr(ctx.config, "max_file_seconds", 0)
    if max_seconds > 0 and (time.perf_counter() - start_time) > max_seconds:
        ctx.file_registry.mark_state(
            file_id=file_id,
            processing_token=processing_token,
            status="failed",
            error="file_timeout",
            source_path=base_path,
            source_key=key,
            content_hash=content_hash,
        )
        raise IngestError(f"File processing timed out: {key}")

    embed_start = time.perf_counter()
    result = embed_texts_batch(ctx, summaries)
    embeddings_by_index: dict[int, list[float]] = result.embeddings_by_index
    embed_elapsed = time.perf_counter() - embed_start
    ctx.logger.debug(
        "FRA embedding batch %s: %.3fs (%d items)",
        key,
        embed_elapsed,
        len(summaries),
    )
    if max_seconds > 0 and (time.perf_counter() - start_time) > max_seconds:
        ctx.file_registry.mark_state(
            file_id=file_id,
            processing_token=processing_token,
            status="failed",
            error="file_timeout",
            source_path=base_path,
            source_key=key,
            content_hash=content_hash,
        )
        raise IngestError(f"File processing timed out: {key}")
    if result.errors_by_index:
        ctx.logger.warning(
            "Embedding batch had %d failures for %s; skipping failed items",
            len(result.errors_by_index),
            key,
        )
        for idx in result.errors_by_index:
            ctx.stats.increment("fra_embeddings_failed")

    added = 0
    for idx, item in enumerate(enriched_items):
        if idx in result.errors_by_index:
            continue
        summary_text = summaries[idx]
        embedding = embeddings_by_index.get(idx)
        if not embedding:
            ctx.logger.warning(
                "Skipping risk item %s due to missing embedding",
                item.get("risk_item_id"),
            )
            continue

        metadata = {
            **sanitise_risk_item_for_metadata(item),
            "source_path": base_path,
            "key": key,
            "source": key,
            "content_hash": content_hash,
            "parsing_confidence": parsing_confidence,
            "parsing_warnings": parsing_warnings,
            "parsing_field_scores": parsing_field_scores,
            "document_type": DocumentTypes.FRA_RISK_ITEM,
        }
        canonical_building = item.get("canonical_building_name")
        if canonical_building:
            metadata["partition_key"] = _fra_partition_key(canonical_building)
        metadata["text"] = summary_text

        valid, reason = validate_with_truncation(
            ctx,
            metadata,
            logger=ctx.logger,
        )
        if not valid:
            ctx.logger.warning(
                "Invalid FRA risk-item metadata for %s: %s",
                item.get("risk_item_id"),
                reason,
            )
            continue

        vectors_to_upsert.append(
            {
                "id": item["risk_item_id"],
                "values": embedding,
                "metadata": metadata,
                "namespace": FRA_RISK_ITEMS_NAMESPACE,
                "_processing_token": processing_token,
            }
        )
        added += 1

    return {
        "added": added,
        "parsing_confidence": parsing_confidence,
        "parsing_warnings": parsing_warnings,
        "parsing_field_scores": parsing_field_scores,
        "missing_action_plan": missing_action_plan,
    }
