#!/usr/bin/env python3
"""
Upsert handling for Alfred Local Ingestion.
This module defines the 
UpsertPolicy, UpsertExecutor, Batcher, and Dispatcher classes, 
which together manage the logic for 
- batching vectors, 
- executing upsert operations with retries and splits, and 
- dispatching to worker threads if configured. 

The VectorWriteCoordinator class serves as a high-level interface for adding vectors 
and ensuring they are processed according to the defined policies.

The UpsertPolicy class encapsulates the decision logic for 
when to retry, split, or fail a batch after an upsert failure, 
based on the type of error, retry count, split depth, and batch size. 

The UpsertExecutor class handles the actual execution of upsert attempts and records relevant metrics. 

The Batcher class manages the buffering and grouping of vectors into batches ready for upsert. 

The Dispatcher class abstracts the logic for either enqueuing batches for worker processing 
or executing them inline, depending on configuration.

This modular design allows for clean separation of concerns, making the code easier to maintain and test.
"""
import time
import threading
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import Any, TYPE_CHECKING
from alfred_exceptions import (
    ExternalServiceError,
    RollbackError,
    RetriableError,
)
from config import (
    INGEST_BACKOFF_BASE,
    INGEST_BACKOFF_CAP,
    INGEST_BACKOFF_JITTER_MIN,
    INGEST_BACKOFF_JITTER_SPAN,
    INGEST_RETRY_ATTEMPTS,
    INGEST_UPSERT_SPLIT_MAX_DEPTH,
    INGEST_UPSERT_SPLIT_MIN_BATCH_SIZE,
)

from .helpers import (
    UpsertQueueItem,
    _estimate_batch_bytes,
    _estimate_metadata_bytes_per_vector,
    _summarise_batch_namespaces,
    _is_rate_limit_error,
)
from .transaction import upsert_vectors_atomic
from .document_processor import Writer
from .utils import (
    IngestionProgressTracker,
)
if TYPE_CHECKING:
    from .context import IngestContext


# ---------------------------------------------------------------------------
# 1. UpsertPolicy  (pure — no I/O, no sleeping, no side effects)
# ---------------------------------------------------------------------------


class _UpsertAction(Enum):
    RETRY = auto()
    SPLIT = auto()
    FAIL = auto()


@dataclass(frozen=True)
class _RetryAction:
    kind: _UpsertAction = _UpsertAction.RETRY


@dataclass(frozen=True)
class _SplitAction:
    kind: _UpsertAction = _UpsertAction.SPLIT


@dataclass(frozen=True)
class _FailAction:
    reason: str
    kind: _UpsertAction = _UpsertAction.FAIL


_NextAction = _RetryAction | _SplitAction | _FailAction


class UpsertPolicy:
    """
    Pure policy object that encodes all retry/split/fail decisions for
    the upsert path.  Contains no I/O — callers are responsible for
    sleeping, queueing, and marking state.
    """

    def __init__(self, *, rng: random.Random) -> None:
        self._rng = rng

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        if isinstance(error, (ExternalServiceError, RetriableError)):
            return True
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        return False

    def backoff_seconds(self, retry_index: int) -> float:
        """Calculate backoff for retry attempt (0-indexed)."""
        exp = min(INGEST_BACKOFF_CAP, INGEST_BACKOFF_BASE * (2 ** retry_index))
        jitter = INGEST_BACKOFF_JITTER_MIN + \
            self._rng.random() * INGEST_BACKOFF_JITTER_SPAN
        return min(INGEST_BACKOFF_CAP, exp + jitter)

    @staticmethod
    def next_action(
        error: Exception,
        retry_index: int,
        split_depth: int,
        batch_size: int,
    ) -> _NextAction:
        """
        Decide what to do after a failed upsert attempt.

        Args:
            error: The exception that occurred during upsert.
            retry_index: Number of retries so far (0 = first failure, 1 = first retry, etc.)
            split_depth: Current recursion depth for batch splitting.
            batch_size: Number of vectors in the current batch.

        Returns:
            One of: _RetryAction, _SplitAction, _FailAction.
        """
        if isinstance(error, RollbackError):
            return _FailAction(reason="rollback_failure")

        if not UpsertPolicy.is_retryable(error):
            return _FailAction(reason="upsert_failed")

        # Retryable and we have retries left (retry_index < max means we can retry).
        # NOTE: INGEST_RETRY_ATTEMPTS is "number of retries" after the initial attempt.
        if retry_index < INGEST_RETRY_ATTEMPTS:
            return _RetryAction()

        # Retries exhausted — try splitting
        if batch_size <= 1:
            return _FailAction(reason="upsert_split_min_batch_reached")

        if split_depth >= INGEST_UPSERT_SPLIT_MAX_DEPTH:
            return _FailAction(reason="upsert_split_depth_exceeded")

        if batch_size <= INGEST_UPSERT_SPLIT_MIN_BATCH_SIZE:
            return _FailAction(reason="upsert_split_min_batch_reached")

        mid = batch_size // 2
        if mid < INGEST_UPSERT_SPLIT_MIN_BATCH_SIZE:
            return _FailAction(reason="upsert_split_min_batch_reached")

        return _SplitAction()


# ---------------------------------------------------------------------------
# 3. UpsertExecutor  (effects — one attempt, no sleeping, no re-queueing)
# ---------------------------------------------------------------------------

class UpsertExecutor:
    """
    Executes a single upsert attempt and records metrics.
    Does NOT sleep, does NOT re-queue batches — that's the caller's job.
    """

    def __init__(self, ctx: "IngestContext", writer: Writer) -> None:
        self._ctx = ctx
        self._writer = writer

    def execute_once(
        self,
        batch: list[dict[str, Any]],
        *,
        retry_index: int = 0,
    ) -> None:
        """
        Write *batch* to the vector store exactly once.
        Emits timing/byte metrics and batch-state records.
        Raises on failure; caller decides retry/split/fail.

        Args:
            batch: List of vector dictionaries to upsert.
            retry_index: Number of retries so far (0 = first try)
        """
        batch_bytes_est = _estimate_batch_bytes(batch)
        batch_metadata_avg = _estimate_metadata_bytes_per_vector(batch)

        upsert_start = time.perf_counter()
        self._writer.write_batch(batch)
        upsert_elapsed = time.perf_counter() - upsert_start

        self._emit_success_metrics(
            batch, upsert_elapsed, batch_bytes_est, batch_metadata_avg, retry_index
        )

    # -- private helpers --

    def _emit_success_metrics(
        self,
        batch: list[dict[str, Any]],
        elapsed: float,
        bytes_est: int | None,
        metadata_avg: int | None,
        retry_index: int,
    ) -> None:
        stats = self._ctx.stats
        stats.observe_timing("upsert_batch_seconds", elapsed)
        stats.increment("upsert_batches_total")
        if bytes_est is not None:
            stats.observe_histogram("upsert_batch_bytes", bytes_est)
        if metadata_avg is not None:
            stats.observe_histogram(
                "upsert_batch_metadata_bytes_per_vector", metadata_avg)
        self._ctx.logger.debug(
            "Upsert batch: %.3fs (%d vectors, ~%d bytes, ~%d metadata bytes/vector,"
            " retry=%d, namespaces=%s)",
            elapsed,
            len(batch),
            bytes_est or 0,
            metadata_avg or 0,
            retry_index,
            _summarise_batch_namespaces(batch),
        )

    def record_batch_state(
        self,
        batch: list[dict[str, Any]],
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        _mark_batch_state(self._ctx, batch, status=status, error=error)

    def record_ingested_files(
        self,
        batch: list[dict[str, Any]],
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        try:
            _record_ingested_files(
                self._ctx, batch, status=status, error=error)
        except Exception as err:  # pylint: disable=broad-except
            self._ctx.logger.warning("FileRegistry update failed: %s", err)


# ---------------------------------------------------------------------------
# 4. Batcher  (pure — buffer/group/flush decisions)
# ---------------------------------------------------------------------------

class Batcher:
    """
    Pure batching logic: groups vectors by namespace and decides when to flush.
    No I/O. Returns ready batches; caller submits them.
    """

    def __init__(self, *, batch_size: int, max_pending: int, flush_seconds: float) -> None:
        self._batch_size = max(1, batch_size)
        self._max_pending = max(1, max_pending)
        self._flush_seconds = max(0.0, flush_seconds)
        self._buffers: dict[str | None, list[dict[str, Any]]] = {}
        self._pending = 0
        self._last_flush = time.perf_counter()

    # -- public API --

    def add(
        self,
        vectors: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """
        Buffer *vectors* and return any batches that are ready to submit.
        Large batches (> max_pending) are split and returned immediately.
        """
        if not vectors:
            return []

        ready: list[list[dict[str, Any]]] = []

        if len(vectors) > self._max_pending:
            ready.extend(self.flush_all())
            ready.extend(self._split_large(vectors))
            return ready

        if self._pending + len(vectors) > self._max_pending:
            ready.extend(self.flush_all())

        for namespace, group in self._group_by_namespace(vectors).items():
            buf = self._buffers.setdefault(namespace, [])
            buf.extend(group)
            self._pending += len(group)
            ready.extend(self._drain_ready(namespace))

        if self._flush_seconds:
            now = time.perf_counter()
            if now - self._last_flush >= self._flush_seconds:
                ready.extend(self.flush_all())

        return ready

    def flush_all(self) -> list[list[dict[str, Any]]]:
        """Flush every namespace buffer and return all ready batches."""
        ready: list[list[dict[str, Any]]] = []
        for namespace in list(self._buffers.keys()):
            ready.extend(self._flush_namespace(namespace))
        self._last_flush = time.perf_counter()
        return ready

    def drain_all(self) -> list[dict[str, Any]]:
        """Return all buffered vectors without batching (used on error paths)."""
        pending: list[dict[str, Any]] = []
        for namespace in list(self._buffers.keys()):
            pending.extend(self._buffers.pop(namespace, []))
        self._pending = 0
        return pending

    # -- private helpers --

    @staticmethod
    def _group_by_namespace(
        vectors: list[dict[str, Any]],
    ) -> dict[str | None, list[dict[str, Any]]]:
        grouped: dict[str | None, list[dict[str, Any]]] = {}
        for v in vectors:
            grouped.setdefault(v.get("namespace"), []).append(v)
        return grouped

    def _drain_ready(self, namespace: str | None) -> list[list[dict[str, Any]]]:
        ready: list[list[dict[str, Any]]] = []
        buf = self._buffers.get(namespace)
        if not buf:
            return ready
        while len(buf) >= self._batch_size:
            batch = buf[: self._batch_size]
            del buf[: self._batch_size]
            self._pending -= len(batch)
            ready.append(batch)
        if not buf:
            self._buffers.pop(namespace, None)
        return ready

    def _flush_namespace(self, namespace: str | None) -> list[list[dict[str, Any]]]:
        ready: list[list[dict[str, Any]]] = []
        buf = self._buffers.pop(namespace, [])
        if not buf:
            return ready
        self._pending -= len(buf)
        for i in range(0, len(buf), self._batch_size):
            ready.append(buf[i: i + self._batch_size])
        return ready

    def _split_large(
        self,
        vectors: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        ready: list[list[dict[str, Any]]] = []
        for group in self._group_by_namespace(vectors).values():
            for i in range(0, len(group), self._batch_size):
                ready.append(group[i: i + self._batch_size])
        return ready


# ---------------------------------------------------------------------------
# 5. Dispatcher  (effects — queue put + worker lifecycle)
# ---------------------------------------------------------------------------

class Dispatcher:
    """
    Handles async dispatch to the upsert worker queue or synchronous
    inline execution.  Owns worker thread lifecycle hooks.
    """

    def __init__(
        self,
        ctx: "IngestContext",
        *,
        writer: Writer,
        use_worker: bool,
        upsert_queue: Queue[UpsertQueueItem | None] | None,
    ) -> None:
        self._ctx = ctx
        self._writer = writer
        self._use_worker = use_worker
        self._upsert_queue = upsert_queue

    def submit(
        self,
        batch: list[dict[str, Any]],
        *,
        progress: IngestionProgressTracker | None = None,
    ) -> None:
        """Dispatch *batch* — either enqueue for the worker or execute inline."""
        if not batch or getattr(self._ctx.config, "dry_run", False):
            return

        if self._use_worker and self._upsert_queue is not None:
            self._enqueue(batch)
            return

        self._execute_inline(batch, progress=progress)

    def _enqueue(self, batch: list[dict[str, Any]]) -> None:
        if self._upsert_queue is None:
            return
        queue_start = time.perf_counter()
        self._upsert_queue.put((batch, 0, 0), block=True)
        queue_wait = time.perf_counter() - queue_start
        self._ctx.stats.observe_timing(
            "upsert_queue_put_wait_seconds", queue_wait)

    def _execute_inline(
        self,
        batch: list[dict[str, Any]],
        *,
        progress: IngestionProgressTracker | None = None,
    ) -> None:
        if progress:
            progress.write_message(
                f"Upserting batch of {len(batch)} vectors...")

        executor = UpsertExecutor(self._ctx, self._writer)
        policy = UpsertPolicy(rng=random.Random())

        retry_index = 0
        split_depth = 0

        while True:
            upsert_start = time.perf_counter()
            try:
                executor.execute_once(batch, retry_index=retry_index)
                return  # Success
            except Exception as error:  # pylint: disable=broad-except
                elapsed = time.perf_counter() - upsert_start
                self._ctx.logger.warning(
                    "Inline upsert batch failed after %.3fs (%d vectors, retry=%d, namespaces=%s): %s",
                    elapsed,
                    len(batch),
                    retry_index,
                    _summarise_batch_namespaces(batch),
                    error,
                )

                action = policy.next_action(
                    error, retry_index, split_depth, len(batch))

                if isinstance(action, _FailAction):
                    self._ctx.logger.exception(
                        "Inline upsert failed: %s", error)
                    self._ctx.stats.increment("upsert_batch_failures_total")
                    _mark_batch_failed(self._ctx, batch, reason=action.reason)
                    raise

                elif isinstance(action, _RetryAction):
                    if _is_rate_limit_error(error):
                        self._ctx.stats.increment(
                            "upsert_throttle_events_total")
                        self._ctx.logger.warning(
                            "Rate limit/throttle detected in inline retry %d",
                            retry_index + 1,
                        )

                    self._ctx.logger.warning(
                        "Inline upsert failed (retry %d/%d); retrying: %s",
                        retry_index + 1,
                        INGEST_RETRY_ATTEMPTS,
                        error,
                    )
                    self._ctx.stats.increment("upsert_batch_retries_total")
                    sleep_secs = policy.backoff_seconds(retry_index)
                    self._ctx.stats.observe_timing(
                        "upsert_backoff_sleep_seconds", sleep_secs)
                    time.sleep(sleep_secs)
                    retry_index += 1

                elif isinstance(action, _SplitAction):
                    self._ctx.logger.warning(
                        "Inline upsert failed after retries; splitting batch of %d",
                        len(batch),
                    )
                    self._ctx.stats.increment("upsert_batch_splits_total")
                    mid = len(batch) // 2
                    # Recursively process splits
                    self._execute_inline(batch[:mid], progress=progress)
                    self._execute_inline(batch[mid:], progress=progress)
                    return


class VectorWriteCoordinator:
    """Single coordinator for batching + upsert scheduling."""

    def __init__(
        self,
        ctx: "IngestContext",
        *,
        writer: Writer,
        use_worker: bool,
        upsert_queue: Queue[UpsertQueueItem | None] | None,
        stop_event: threading.Event,
        max_pending_vectors: int,
        upsert_batch: int,
        flush_seconds: float,
    ) -> None:
        self._ctx = ctx
        self._stop_event = stop_event
        self._batcher = Batcher(
            batch_size=max(
                1, min(int(upsert_batch), max(1, int(max_pending_vectors)))),
            max_pending=max(1, int(max_pending_vectors)),
            flush_seconds=max(0.0, float(flush_seconds)),
        )
        self._dispatcher = Dispatcher(
            ctx,
            writer=writer,
            use_worker=use_worker,
            upsert_queue=upsert_queue,
        )

    def add_vectors(
        self,
        vectors: list[dict[str, Any]],
        *,
        progress: IngestionProgressTracker | None = None,
    ) -> None:
        if not vectors or self._stop_event.is_set():
            return
        for batch in self._batcher.add(vectors):
            self._dispatcher.submit(batch, progress=progress)

    def flush_all(self, *, progress: IngestionProgressTracker | None = None) -> None:
        for batch in self._batcher.flush_all():
            self._dispatcher.submit(batch, progress=progress)

    def close(self, *, progress: IngestionProgressTracker | None = None) -> None:
        self.flush_all(progress=progress)

    def drain_pending_vectors(self) -> list[dict[str, Any]]:
        return self._batcher.drain_all()


# Add this function before the UpsertExecutor class (around line 160-170)

def _mark_batch_state(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    status: str,
    error: str | None = None,
) -> None:
    """
    Record batch state in context stats/logs.

    Args:
        ctx: Ingest context
        batch: Batch of vectors
        status: Status string (e.g., 'success', 'failed', 'retrying')
        error: Optional error message
    """
    ctx.logger.info(
        "Batch state: status=%s, vectors=%d, namespaces=%s, error=%s",
        status,
        len(batch),
        _summarise_batch_namespaces(batch),
        error or "none",
    )
    ctx.stats.increment(f"batch_state_{status}_total")


def _mark_batch_failed(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    reason: str,
) -> None:
    """Mark a batch as failed with specific reason."""
    _mark_batch_state(ctx, batch, status="failed", error=reason)


def _record_ingested_files(
    ctx: "IngestContext",
    batch: list[dict[str, Any]],
    *,
    status: str,
    error: str | None = None,
) -> None:
    """
    Record file ingestion status in file registry.

    Args:
        ctx: Ingest context
        batch: Batch of vectors
        status: Status string
        error: Optional error message
    """
    # Extract unique file IDs from batch metadata
    file_ids = set()
    for vector in batch:
        metadata = vector.get("metadata", {})
        if "file_id" in metadata:
            file_ids.add(metadata["file_id"])

    # Update file registry if available
    if hasattr(ctx, "file_registry") and ctx.file_registry:
        for file_id in file_ids:
            try:
                ctx.file_registry.mark_state(
                    file_id=file_id, processing_token="", status=status, error=error)
            except Exception as err:  # pylint: disable=broad-except
                ctx.logger.debug(
                    "Failed to update file registry for %s: %s", file_id, err)

# ---------------------------------------------------------------------------
# Upsert worker  (Delegates to UpsertPolicy + UpsertExecutor)
# ---------------------------------------------------------------------------


def _upsert_worker(
    ctx: "IngestContext",
    upsert_queue: Queue[UpsertQueueItem | None],
    stop_event: threading.Event,
    errors: list[Exception],
) -> None:
    writer = Writer(ctx, upsert_vectors_atomic)
    executor = UpsertExecutor(ctx, writer)
    policy = UpsertPolicy(rng=random.Random())

    while True:
        queued: UpsertQueueItem | None = upsert_queue.get()
        batch: list[dict[str, Any]] = []
        retry_index = 0
        split_depth = 0
        upsert_start: float = 0.0

        try:
            if queued is None:
                # Sentinel received — stop processing
                return

            batch, retry_index, split_depth = queued

            # Process all queued batches, even if stop_event is set
            # (they were queued before shutdown)
            upsert_start = time.perf_counter()
            executor.execute_once(batch, retry_index=retry_index)
            upsert_elapsed = time.perf_counter() - upsert_start
            ctx.logger.info(
                "Upsert batch complete: %.3fs (%d vectors, retry=%d, namespaces=%s)",
                upsert_elapsed,
                len(batch),
                retry_index,
                _summarise_batch_namespaces(batch),
            )

        except Exception as error:  # pylint: disable=broad-except
            upsert_elapsed = time.perf_counter() - upsert_start
            ctx.logger.warning(
                "Upsert batch failed after %.3fs (%d vectors, retry=%d,"
                " namespaces=%s): %s",
                upsert_elapsed,
                len(batch),
                retry_index,
                _summarise_batch_namespaces(batch),
                error,
            )

            action = policy.next_action(
                error, retry_index, split_depth, len(batch))

            if isinstance(action, _FailAction):
                if action.reason == "rollback_failure":
                    ctx.logger.critical(
                        "Upsert worker rollback failed: %s", error)
                    try:
                        ctx.stats.increment("rollback_failures_total")
                        ctx.event_sink.emit_event(
                            {
                                "event_type": "rollback_failure",
                                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                                "error": str(error),
                            }
                        )
                    except Exception as alert_error:  # pylint: disable=broad-except
                        ctx.logger.warning(
                            "Rollback alert emission failed: %s", alert_error)
                    errors.append(error)
                    stop_event.set()
                else:
                    ctx.logger.exception("Upsert worker failed: %s", error)
                    ctx.stats.increment("upsert_batch_failures_total")
                    _mark_batch_failed(ctx, batch, reason=action.reason)

            elif isinstance(action, _RetryAction):
                # Don't retry if stop_event is set (shutdown in progress)
                if stop_event.is_set():
                    ctx.logger.warning(
                        "Shutdown in progress; failing batch instead of retrying"
                    )
                    _mark_batch_failed(
                        ctx, batch, reason="shutdown_during_retry")
                else:
                    # Check for rate limiting / throttling
                    if _is_rate_limit_error(error):
                        ctx.stats.increment("upsert_throttle_events_total")
                        ctx.logger.warning(
                            "Rate limit/throttle detected in batch retry %d",
                            retry_index + 1,
                        )

                    ctx.logger.warning(
                        "Upsert batch failed (retry %d/%d); retrying: %s",
                        retry_index + 1,
                        INGEST_RETRY_ATTEMPTS,
                        error,
                    )
                    ctx.stats.increment("upsert_batch_retries_total")
                    sleep_secs = policy.backoff_seconds(retry_index)
                    ctx.stats.observe_timing(
                        "upsert_backoff_sleep_seconds", sleep_secs)
                    time.sleep(sleep_secs)
                    upsert_queue.put((batch, retry_index + 1, split_depth))

            elif isinstance(action, _SplitAction):
                # Don't split if stop_event is set (shutdown in progress)
                if stop_event.is_set():
                    ctx.logger.warning(
                        "Shutdown in progress; failing batch instead of splitting"
                    )
                    _mark_batch_failed(
                        ctx, batch, reason="shutdown_during_split")
                else:
                    ctx.logger.warning(
                        "Upsert batch failed after retries; splitting batch of %d",
                        len(batch),
                    )
                    ctx.stats.increment("upsert_batch_splits_total")
                    mid = len(batch) // 2
                    upsert_queue.put((batch[:mid], 0, split_depth + 1))
                    upsert_queue.put((batch[mid:], 0, split_depth + 1))

        finally:
            upsert_queue.task_done()
