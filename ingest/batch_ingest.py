#!/usr/bin/env python3
"""
Batch Ingest for AskAlfred.
Core logic for batch ingestion, focused on local directory ingestion with progress tracking and security.
This module contains two main layers:
1. run_ingest() - Core ingest loop, independent of UI/progress concerns. Processes files, handles worker coordination, and returns an IngestReport.
2. ingest_local_directory_with_progress() - Thin wrapper around run_ingest() that handles local file discovery, building resolution, and progress tracking. 
Also emits metrics and logs summary after completion.
"""
import time
import threading
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alfred_exceptions import (
    IngestError,
)
from config import (
    INGEST_UPSERT_JOIN_TIMEOUT_SECONDS,
    INGEST_UPSERT_JOIN_POLL_SECONDS,
)
from filename_building_parser import FilenameBuildingResolver
from .context import IngestContext
from .document_content import (
    load_building_names_with_aliases,
)
from .document_processor import DocumentProcessor, FileIngestOrchestrator, Writer
from .utils import (
    IngestionProgressTracker,
    list_local_files_secure,
)
from .helpers import UpsertQueueItem
from .upsert_handler import (
    VectorWriteCoordinator, _mark_batch_failed,
)
from .transaction import (
    extract_fra_risk_items_integration, upsert_vectors_atomic,)
from .upsert_handler import (_upsert_worker)


# ---------------------------------------------------------------------------
# 7. IngestReport  +  run_ingest()  (core logic, no UI)
# ---------------------------------------------------------------------------

@dataclass
class IngestReport:
    """Value object returned by run_ingest()."""
    files_found: int
    files_processed: int
    files_skipped: int
    files_failed: int
    total_vectors: int
    duration_seconds: float
    failed_files: list[str] = field(default_factory=list)

    @property
    def vectors_per_second(self) -> float:
        return self.total_vectors / self.duration_seconds if self.duration_seconds > 0 else 0.0


def _run_ingest_sequential(
    ctx: IngestContext,
    objs: list[dict[str, Any]],
    orchestrator: FileIngestOrchestrator,
    coordinator: VectorWriteCoordinator,
    upsert_stop_event: threading.Event,
    progress: IngestionProgressTracker | None = None,
) -> None:
    """Process files sequentially."""
    for obj in objs:
        if upsert_stop_event.is_set():
            break
        filename = obj.get("Key", "")
        try:
            result = orchestrator.process(obj)
            if result.vectors:
                coordinator.add_vectors(result.vectors, progress=progress)
            if progress:
                progress.update(
                    filename,
                    vectors=result.vector_count,
                    status=result.status,
                )
            ctx.stats.increment("files_processed")
            if result.status == "skipped":
                ctx.stats.increment("files_skipped")
        except Exception as error:  # pylint: disable=broad-except
            ctx.logger.warning("Failed to ingest file %s: %s",
                               obj.get("Key"), error)
            ctx.stats.increment("files_failed")
            failed_key = obj.get("Key") or obj.get("key") or ""
            if failed_key:
                ctx.stats.append_failed(failed_key)
            ctx.stats.increment("files_processed")
            if progress:
                progress.update(filename, status="failed")


def _run_ingest_parallel(
    ctx: IngestContext,
    objs: list[dict[str, Any]],
    orchestrator: FileIngestOrchestrator,
    coordinator: VectorWriteCoordinator,
    upsert_stop_event: threading.Event,
    progress: IngestionProgressTracker | None = None,
) -> None:
    """Process files in parallel using thread pool."""
    worker_count = max(
        ctx.config.max_io_workers,
        ctx.config.max_parse_workers,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(orchestrator.process, obj): obj
            for obj in objs
        }
        for future in as_completed(futures):
            if upsert_stop_event.is_set():
                break
            obj = futures[future]
            filename = obj.get("Key", "")
            try:
                result = future.result()
                if result.vectors:
                    coordinator.add_vectors(result.vectors, progress=progress)
                if progress:
                    progress.update(
                        filename,
                        vectors=result.vector_count,
                        status=result.status,
                    )
                ctx.stats.increment("files_processed")
                if result.status == "skipped":
                    ctx.stats.increment("files_skipped")
            except Exception as error:  # pylint: disable=broad-except
                ctx.logger.warning("Failed to ingest file %s: %s",
                                   obj.get("Key"), error)
                ctx.stats.increment("files_failed")
                failed_key = obj.get("Key") or obj.get("key") or ""
                if failed_key:
                    ctx.stats.append_failed(failed_key)
                ctx.stats.increment("files_processed")
                if progress:
                    progress.update(filename, status="failed")


def run_ingest(
    ctx: IngestContext,
    objs: list[dict[str, Any]],
    *,
    orchestrator: FileIngestOrchestrator,
    coordinator: VectorWriteCoordinator,
    upsert_stop_event: threading.Event,
    upsert_queue: Queue[UpsertQueueItem | None] | None,
    upsert_threads: list[threading.Thread] | None,
    upsert_errors: list[Exception],
    use_worker: bool,
    base_path: str,
    progress: IngestionProgressTracker | None = None,
) -> IngestReport:
    """
    Core ingest loop — no progress bars, no UI concerns.

    Iterates *objs*, feeds vectors into *coordinator*, and handles
    worker teardown.  Returns an IngestReport.

    Progress updates are delegated to *progress* if provided.
    """
    t_start = time.time()

    if ctx.config.max_io_workers == 1 and ctx.config.max_parse_workers == 1:
        _run_ingest_sequential(ctx, objs, orchestrator,
                               coordinator, upsert_stop_event, progress)
    else:
        _run_ingest_parallel(ctx, objs, orchestrator,
                             coordinator, upsert_stop_event, progress)

    # Close coordinator and handle abort/normal paths
    if upsert_stop_event.is_set():
        # Abort path: drain pending work and mark failed
        pending = coordinator.drain_pending_vectors()
        _mark_batch_failed(ctx, pending, reason="upsert_worker_failed")
    else:
        # Normal path: flush all batches to queue/inline
        coordinator.close(progress=progress)

    # Tear down the worker thread (coordinator already closed)
    if use_worker and upsert_queue is not None and upsert_threads:
        _teardown_worker(
            ctx,
            upsert_stop_event=upsert_stop_event,
            upsert_queue=upsert_queue,
            upsert_threads=upsert_threads,
        )

    # Check for worker errors
    if upsert_errors:
        ctx.logger.error("Upsert worker reported %d error(s)",
                         len(upsert_errors))
        for error in upsert_errors:
            ctx.logger.error("  - %s", error)

    stats = ctx.stats.get_stats()
    duration = time.time() - t_start
    return IngestReport(
        files_found=len(objs),
        files_processed=stats["files_processed"],
        files_skipped=stats["files_skipped"],
        files_failed=stats["files_failed"],
        total_vectors=stats["total_vectors"],
        duration_seconds=duration,
        failed_files=list(stats.get("failed_files", [])),
    )


def _teardown_worker(
    ctx: IngestContext,
    *,
    upsert_stop_event: threading.Event,
    upsert_queue: Queue[UpsertQueueItem | None],
    upsert_threads: list[threading.Thread],
) -> None:
    """
    Tear down the upsert worker thread.

    Two paths:
    - Normal: drain queue gracefully, then stop worker
    - Abort: drain and fail all pending work, then stop worker

    NOTE: Coordinator must be closed BEFORE calling this function.
    """
    if upsert_stop_event.is_set():
        # Abort path: drain queue and mark all pending batches as failed
        ctx.logger.warning(
            "Upsert worker stop_event set; draining and failing pending batches")
        pending_count = 0
        while True:
            try:
                queued = upsert_queue.get_nowait()
                if queued is None:
                    # Sentinel already queued (unlikely), put it back
                    upsert_queue.put(None)
                    break
                batch, retry_index, split_depth = queued
                _mark_batch_failed(ctx, batch, reason="worker_aborted")
                upsert_queue.task_done()
                pending_count += 1
            except Empty:
                break

        if pending_count > 0:
            ctx.logger.warning(
                "Marked %d pending batch(es) as failed during abort", pending_count)

        # Send sentinel and wait for worker to exit
        for _ in upsert_threads:
            upsert_queue.put(None)
        for thread in upsert_threads:
            thread.join(timeout=INGEST_UPSERT_JOIN_TIMEOUT_SECONDS)

    else:
        # Normal path: wait for all work to complete, then stop worker
        ctx.logger.info("Waiting for upsert queue to drain...")
        queue_join_start = time.perf_counter()

        # Poll join with timeout to allow logging
        queue_drained = False
        elapsed = 0.0
        while elapsed < INGEST_UPSERT_JOIN_TIMEOUT_SECONDS:
            try:
                # Try to join with a short timeout so we can log progress
                remaining = INGEST_UPSERT_JOIN_TIMEOUT_SECONDS - elapsed
                poll_timeout = min(INGEST_UPSERT_JOIN_POLL_SECONDS, remaining)

                # Note: queue.join() doesn't take timeout in Python, so we check size
                if upsert_queue.unfinished_tasks == 0:
                    queue_drained = True
                    break

                time.sleep(poll_timeout)
                elapsed = time.perf_counter() - queue_join_start

                # Log every 10s after first 10s
                if elapsed > 10.0 and int(elapsed) % 10 == 0:
                    ctx.logger.info(
                        "Still waiting for upsert queue (%.1fs elapsed, %d tasks remaining)...",
                        elapsed,
                        upsert_queue.unfinished_tasks
                    )
            except KeyboardInterrupt:
                ctx.logger.warning(
                    "Keyboard interrupt during queue drain; aborting")
                upsert_stop_event.set()
                # Switch to abort path (don't recurse, handle inline)
                while True:
                    try:
                        queued = upsert_queue.get_nowait()
                        if queued is None:
                            upsert_queue.put(None)
                            break
                        batch, _, _ = queued
                        _mark_batch_failed(
                            ctx, batch, reason="keyboard_interrupt")
                        upsert_queue.task_done()
                    except Empty:
                        break
                for _ in upsert_threads:
                    upsert_queue.put(None)
                for thread in upsert_threads:
                    thread.join(timeout=INGEST_UPSERT_JOIN_TIMEOUT_SECONDS)
                return

        if not queue_drained:
            ctx.logger.warning(
                "Upsert queue join timed out after %.1fs (%d tasks remaining); sending sentinel anyway",
                INGEST_UPSERT_JOIN_TIMEOUT_SECONDS,
                upsert_queue.unfinished_tasks
            )
            ctx.logger.warning(
                "Upsert queue timeout details: unfinished_tasks=%d",
                upsert_queue.unfinished_tasks,
            )
            upsert_stop_event.set()
            drained = 0
            failed_files: set[str] = set()
            while True:
                try:
                    queued = upsert_queue.get_nowait()
                    if queued is None:
                        # Sentinel already queued (unlikely), put it back
                        upsert_queue.put(None)
                        break
                    batch, _, _ = queued
                    _mark_batch_failed(
                        ctx, batch, reason="queue_drain_timeout")
                    for vector in batch:
                        metadata = vector.get("metadata") or {}
                        file_key = metadata.get(
                            "key") or metadata.get("source")
                        if isinstance(file_key, str) and file_key:
                            failed_files.add(file_key)
                            continue
                        vector_id = vector.get("id")
                        if isinstance(vector_id, str) and vector_id:
                            file_id = vector_id.split(":", 1)[0]
                            if file_id:
                                failed_files.add(file_id)
                    upsert_queue.task_done()
                    drained += 1
                except Empty:
                    break
            if failed_files:
                ctx.stats.increment("files_failed", len(failed_files))
                for failed_file in sorted(failed_files):
                    ctx.stats.append_failed(failed_file)
                preview = ", ".join(sorted(failed_files)[:10])
                ctx.logger.warning(
                    "Upsert queue timeout pending files (showing up to 10 of %d): %s",
                    len(failed_files),
                    preview,
                )
            if drained > 0:
                ctx.logger.warning(
                    "Drained and failed %d pending batch(es) after queue timeout",
                    drained,
                )
        else:
            queue_join_elapsed = time.perf_counter() - queue_join_start
            ctx.logger.info("Upsert queue drained in %.3fs",
                            queue_join_elapsed)

        # Send sentinel to stop worker
        for _ in upsert_threads:
            upsert_queue.put(None)

        # Wait for thread to exit
        ctx.logger.info("Waiting for upsert worker thread to exit...")
        thread_join_start = time.perf_counter()
        for thread in upsert_threads:
            thread.join(timeout=INGEST_UPSERT_JOIN_TIMEOUT_SECONDS)
        thread_join_elapsed = time.perf_counter() - thread_join_start

        still_alive = [t.name for t in upsert_threads if t.is_alive()]
        if still_alive:
            ctx.logger.error(
                "Upsert worker thread(s) did not exit after %.1fs: %s",
                INGEST_UPSERT_JOIN_TIMEOUT_SECONDS,
                ", ".join(still_alive),
            )
        else:
            ctx.logger.info(
                "Upsert worker thread(s) exited in %.3fs", thread_join_elapsed)


# ---------------------------------------------------------------------------
# 8. ingest_local_directory_with_progress  (thin UI/progress wrapper)
# ---------------------------------------------------------------------------

def _emit_ingest_metrics(
    ctx: IngestContext,
    report: IngestReport,
    base_path: str,
) -> None:
    """Export Prometheus and event-sink metrics after ingestion."""
    prom_path = (
        getattr(ctx.config, "prometheus_metrics_file", "") or "").strip()
    if prom_path:
        try:
            ctx.event_sink.export_metrics(
                stats=ctx.stats,
                output_path=prom_path,
                duration_seconds=report.duration_seconds,
                vectors_per_second=report.vectors_per_second,
                source_path=base_path,
                dry_run=bool(getattr(ctx.config, "dry_run", False)),
                upsert_workers=ctx.config.upsert_workers,
            )
            ctx.logger.info("📊 Exported Prometheus metrics to %s", prom_path)
        except IngestError as error:
            ctx.logger.warning(
                "Could not export Prometheus metrics: %s", error)

    if getattr(ctx.config, "export_events", False):
        metrics = {
            "event_type": "ingestion_summary",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "source_path": base_path,
            "dry_run": bool(getattr(ctx.config, "dry_run", False)),
            "duration_seconds": report.duration_seconds,
            "files_processed": report.files_processed,
            "vectors_created": report.total_vectors,
            "vectors_per_second": report.vectors_per_second,
            "failures": report.files_failed,
        }
        try:
            ctx.event_sink.emit_event(metrics)
        except IngestError as error:
            ctx.logger.warning(
                "Could not write ingestion summary event: %s", error)


def _log_ingest_summary(ctx: IngestContext, report: IngestReport) -> None:
    ctx.logger.info(
        """========================================
            INGESTION SUMMARY
            ========================================
            Files found:          %d
            Files processed:      %d
            Files skipped:        %d
            Files failed:         %d
            Total vectors:        %d
            Duration:             %.2fs
            Avg speed:            %.1f vectors/sec
            ========================================
            """,
        report.files_found,
        report.files_processed,
        report.files_skipped,
        report.files_failed,
        report.total_vectors,
        report.duration_seconds,
        report.vectors_per_second,
    )
    if report.failed_files:
        ctx.logger.warning("Failed files:")
        for failed_file in report.failed_files:
            ctx.logger.warning("  - %s", failed_file)
    ctx.logger.info("Ingestion complete")


def ingest_local_directory_with_progress(
    ctx: IngestContext,
    use_progress_bar: bool = True,
) -> None:
    """
    Enhanced ingestion with progress tracking and security.

    This is a thin UI/progress wrapper around run_ingest().
    All core logic lives in run_ingest() and can be exercised independently.
    """
    base_path = ctx.config.local_path

    if not Path(base_path).exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    objs = list_local_files_secure(
        base_path,
        ctx.config.ext_whitelist,
        ctx.config.max_file_mb,
        logger=ctx.logger,
    )
    ctx.logger.info("Found %d files to process in %s", len(objs), base_path)

    if not objs:
        ctx.logger.warning("No files found to process")
        return

    # Building resolution
    name_to_canonical: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}
    known_buildings: list[str] = []
    csv_candidates = [
        obj["Key"]
        for obj in objs
        if "Property" in obj["Key"]
        and obj["Key"].endswith(".csv")
        and "maintenance" not in obj["Key"].lower()
    ]
    if csv_candidates:
        known_buildings, name_to_canonical, alias_to_canonical = (
            load_building_names_with_aliases(ctx, base_path, csv_candidates[0])
        )
    else:
        ctx.logger.warning(
            "No property CSV found for building name resolution")

    building_resolver = FilenameBuildingResolver(
        name_to_canonical=name_to_canonical,
        alias_to_canonical=alias_to_canonical,
        known_buildings=known_buildings,
    )

    processor = DocumentProcessor(
        ctx=ctx,
        base_path=base_path,
        alias_to_canonical=alias_to_canonical,
        fra_vector_extractor=extract_fra_risk_items_integration,
        building_resolver=building_resolver,
    )
    orchestrator = FileIngestOrchestrator(processor)

    use_worker = (
        not getattr(ctx.config, "dry_run", False)
        and ctx.config.upsert_strategy == "worker"
    )
    upsert_queue: Queue[UpsertQueueItem | None] | None = None
    upsert_errors: list[Exception] = []
    upsert_stop_event = threading.Event()
    ctx.upsert_stop_event = upsert_stop_event

    cpu_pool: ProcessPoolExecutor | None = None
    parse_pool: ProcessPoolExecutor | None = None
    try:
        if ctx.config.max_parse_workers > 1:
            cpu_pool = ProcessPoolExecutor(
                max_workers=ctx.config.max_parse_workers
            )
            orchestrator.set_cpu_pool(cpu_pool)

        if ctx.config.max_parse_workers > 1:
            parse_pool = ProcessPoolExecutor(
                max_workers=ctx.config.max_parse_workers
            )
            orchestrator.set_parse_pool(parse_pool)

        upsert_threads: list[threading.Thread] | None = None

        if use_worker:
            batch_size = max(
                1,
                min(int(ctx.config.upsert_batch),
                    max(1, int(ctx.config.max_pending_vectors))),
            )
            queue_max = max(
                1, int(ctx.config.max_pending_vectors) // batch_size)
            upsert_queue = Queue(maxsize=queue_max)
            upsert_threads = []
            worker_count = max(1, int(ctx.config.upsert_workers))
            for idx in range(worker_count):
                thread = threading.Thread(
                    target=_upsert_worker,
                    name=f"upsert-worker-{idx + 1}",
                    args=(ctx, upsert_queue, upsert_stop_event, upsert_errors),
                    daemon=True,
                )
                thread.start()
                upsert_threads.append(thread)

        writer = Writer(ctx, upsert_vectors_atomic)
        coordinator = VectorWriteCoordinator(
            ctx,
            writer=writer,
            use_worker=use_worker,
            upsert_queue=upsert_queue,
            stop_event=upsert_stop_event,
            max_pending_vectors=ctx.config.max_pending_vectors,
            upsert_batch=ctx.config.upsert_batch,
            flush_seconds=getattr(ctx.config, "upsert_flush_seconds", 0),
        )

        with IngestionProgressTracker(
            len(objs),
            use_tqdm=use_progress_bar,
            progress_log_interval=ctx.config.progress_log_interval,
            logger=ctx.logger,
        ) as progress:
            report = run_ingest(
                ctx,
                objs,
                orchestrator=orchestrator,
                coordinator=coordinator,
                upsert_stop_event=upsert_stop_event,
                upsert_queue=upsert_queue,
                upsert_threads=upsert_threads,
                upsert_errors=upsert_errors,
                use_worker=use_worker,
                base_path=base_path,
                progress=progress,
            )

    finally:
        ctx.upsert_stop_event = None
        if cpu_pool is not None:
            cpu_pool.shutdown(wait=True)
        if parse_pool is not None:
            parse_pool.shutdown(wait=True)

    _emit_ingest_metrics(ctx, report, base_path)
    _log_ingest_summary(ctx, report)
