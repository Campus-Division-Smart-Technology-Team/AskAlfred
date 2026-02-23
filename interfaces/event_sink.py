# EventSink port
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Protocol, Mapping, runtime_checkable

from ingest.utils import MetricsExporter


@runtime_checkable
class MetricsReader(Protocol):
    def get_stats(self) -> dict[str, Any]: ...


class EventSink(Protocol):
    def emit_event(self, event: dict[str, Any]) -> None: ...
    def export_metrics(
        self,
        *,
        stats: MetricsReader | Mapping[str, Any],
        output_path: str,
        duration_seconds: float,
        vectors_per_second: float,
        source_path: str,
        dry_run: bool,
        upsert_workers: int | None = None,
    ) -> None: ...


class JsonlPrometheusEventSink:
    """Writes JSONL events and Prometheus metrics."""

    def __init__(
        self,
        *,
        events_path: Optional[str] = None,
        lock: Optional[Lock] = None,
    ):
        self._events_path = (events_path or "").strip() or None
        self._lock = lock or Lock()
        self._metrics = MetricsExporter()

    def emit_event(self, event: dict[str, Any]) -> None:
        if not self._events_path:
            return
        path = Path(self._events_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)

    def export_metrics(
        self,
        *,
        stats: MetricsReader | Mapping[str, Any],
        output_path: str,
        duration_seconds: float,
        vectors_per_second: float,
        source_path: str,
        dry_run: bool,
        upsert_workers: int | None = None,
    ) -> None:
        if isinstance(stats, MetricsReader):
            stats_payload = stats.get_stats()
        else:
            stats_payload = dict(stats)
        self._metrics.export_prometheus(
            stats=stats_payload,
            output_path=output_path,
            duration_seconds=duration_seconds,
            vectors_per_second=vectors_per_second,
            source_path=source_path,
            dry_run=dry_run,
            upsert_workers=upsert_workers,
        )


__all__ = [
    "EventSink",
    "JsonlPrometheusEventSink",
]
