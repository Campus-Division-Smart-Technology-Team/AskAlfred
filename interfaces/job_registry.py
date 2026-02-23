# JobRegistry port
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Optional, Any, cast
from datetime import datetime, timezone

from redis import Redis
from config.constant import (
    INGEST_JOB_TTL_DEFAULT_SECONDS,
    INGEST_JOB_TTL_SUPERSEDE_SECONDS,
)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    job_type: str
    status: str  # "pending" | "success" | "failed" | "partial"
    started_at_iso: str
    finished_at_iso: str | None = None
    error: str | None = None
    meta: dict[str, Any] | None = None


class JobRegistry(Protocol):
    def get(self, job_id: str) -> Optional[JobRecord]: ...
    def try_start(
        self,
        *,
        job_id: str,
        job_type: str,
        status: str = "processing",
        meta: dict[str, Any] | None = None,
    ) -> bool: ...
    def upsert(self, record: JobRecord) -> None: ...
    def delete(self, job_id: str) -> None: ...


class RedisJobRegistry:
    """Redis-backed JobRegistry using JSON-encoded records."""

    def __init__(
        self,
        client: Redis,
        *,
        prefix: str = "ingest:job:",
        ttl_by_type: dict[str, int] | None = None,
        default_ttl_seconds: int = INGEST_JOB_TTL_DEFAULT_SECONDS,
    ):
        self._client = client
        self._prefix = prefix
        self._default_ttl_seconds = int(default_ttl_seconds)
        self._ttl_by_type = ttl_by_type or {
            "fra_supersede": INGEST_JOB_TTL_SUPERSEDE_SECONDS,
            "fra_supersession": INGEST_JOB_TTL_SUPERSEDE_SECONDS,
        }

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def _ttl_for_job_type(self, job_type: str) -> int:
        if not job_type:
            return self._default_ttl_seconds
        return int(self._ttl_by_type.get(job_type, self._default_ttl_seconds))

    def get(self, job_id: str) -> Optional[JobRecord]:
        raw = self._client.get(self._key(job_id))
        if not raw:
            return None
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw_text = raw.decode("utf-8")
            elif isinstance(raw, str):
                raw_text = raw
            else:
                return None
            payload = cast(dict[str, Any], json.loads(raw_text))
        except (TypeError, json.JSONDecodeError):
            return None
        return JobRecord(
            job_id=payload.get("job_id", job_id),
            job_type=payload.get("job_type", ""),
            status=payload.get("status", "unknown"),
            started_at_iso=payload.get("started_at_iso", ""),
            finished_at_iso=payload.get("finished_at_iso"),
            error=payload.get("error"),
            meta=payload.get("meta") or None,
        )

    def try_start(
        self,
        *,
        job_id: str,
        job_type: str,
        status: str = "processing",
        meta: dict[str, Any] | None = None,
    ) -> bool:
        started_at = datetime.now(timezone.utc).isoformat() + "Z"
        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            "started_at_iso": started_at,
            "finished_at_iso": None,
            "error": None,
            "meta": meta or {},
        }
        ttl_seconds = self._ttl_for_job_type(job_type)
        result = self._client.set(
            self._key(job_id),
            json.dumps(payload, ensure_ascii=False),
            nx=True,
            ex=ttl_seconds,
        )
        return bool(result)

    def upsert(self, record: JobRecord) -> None:
        payload = {
            "job_id": record.job_id,
            "job_type": record.job_type,
            "status": record.status,
            "started_at_iso": record.started_at_iso,
            "finished_at_iso": record.finished_at_iso,
            "error": record.error,
            "meta": record.meta or {},
        }
        ttl_seconds = self._ttl_for_job_type(record.job_type)
        self._client.set(
            self._key(record.job_id),
            json.dumps(payload, ensure_ascii=False),
            ex=ttl_seconds,
        )

    def delete(self, job_id: str) -> None:
        self._client.delete(self._key(job_id))


class NoOpJobRegistry:
    """No-op JobRegistry for dry-run mode."""

    def get(self, job_id: str) -> Optional[JobRecord]:
        # pylint: disable=unused-argument
        return None

    def try_start(
        self,
        *,
        job_id: str,
        job_type: str,
        status: str = "processing",
        meta: dict[str, Any] | None = None,
    ) -> bool:
        # pylint: disable=unused-argument
        return True

    def upsert(self, record: JobRecord) -> None:
        # pylint: disable=unused-argument
        return None

    def delete(self, job_id: str) -> None:
        # pylint: disable=unused-argument
        return None


__all__ = [
    "JobRecord",
    "JobRegistry",
    "RedisJobRegistry",
    "NoOpJobRegistry",
]
