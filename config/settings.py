#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for AskAlfred chatbot.
Enhanced with validation and type safety.
"""

from typing import TypedDict, Optional
from dataclasses import dataclass, field
import os
import logging
from pathlib import Path

import streamlit as st

from alfred_exceptions import ConfigError, RoutingError
from .constant import (
    ANSWER_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_NAMESPACE,
    DIMENSION,
    DocumentTypes,
    FRA_RISK_ITEMS_NAMESPACE,
    INGEST_METADATA_MAX_SIZE,
    INGEST_METADATA_MAX_TEXT_TOKENS,
    INGEST_PROGRESS_LOG_INTERVAL,
    INGEST_DEDUP_FETCH_BATCH_SIZE,
    INGEST_UPSERT_FLUSH_SECONDS,
    INGEST_UPSERT_JOIN_TIMEOUT_SECONDS,
    INGEST_UPSERT_JOIN_POLL_SECONDS,
    OPENAI_TIMEOUT_DEFAULT_S,
    OPENAI_CONNECT_TIMEOUT_S,
    OPENAI_READ_TIMEOUT_S,
    OPENAI_WRITE_TIMEOUT_S,
    OPENAI_POOL_TIMEOUT_S,
    INGEST_MAX_FILE_SECONDS,
    INDEX_CONFIGS,
    MIN_SCORE_THRESHOLD,
    NAMESPACE_MAPPINGS,
    RISK_LEVEL_MAP,
    SEARCH_ALL_NAMESPACES,
    TARGET_INDEXES,
    FRA_SUPERSESSION_SINGLE_THREADED,
)


# Try to load a local .env if python-dotenv is available; otherwise, ignore
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pylint: disable=broad-except
    pass

# Streamlit Cloud secrets fallback
try:
    if "PINECONE_API_KEY" not in os.environ and "PINECONE_API_KEY" in st.secrets:
        # type: ignore
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
        # type: ignore
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:  # pylint: disable=broad-except
    pass

# ===========================================================================
# NAMESPACE UTILITIES
# ===========================================================================


def get_display_namespace(namespace: Optional[str]) -> str:
    return "__default__" if namespace is None else namespace


def get_internal_namespace(namespace: Optional[str]) -> Optional[str]:
    if namespace in ("", None):
        return None
    if namespace == "__default__":
        return "__default__"
    return namespace


def normalise_ns(ns: Optional[str]) -> Optional[str]:
    return get_internal_namespace(ns)


def resolve_namespace(doc_type: Optional[str]) -> Optional[str]:
    if not doc_type:
        logging.debug(
            "resolve_namespace: no doc_type supplied, using default namespace.")
        return DEFAULT_NAMESPACE
    try:
        return NAMESPACE_MAPPINGS[doc_type]
    except KeyError as exc:
        raise RoutingError(f"Undefined doc_type: {doc_type}") from exc


def _route_namespace(doc_type: Optional[str]) -> Optional[str]:
    return resolve_namespace(doc_type)


class IndexConfig(TypedDict):
    model: str
    dimension: int


def get_index_config(index_name: str) -> IndexConfig:
    config = INDEX_CONFIGS.get(index_name)
    if config:
        return IndexConfig(
            model=config.get("model", DEFAULT_EMBED_MODEL),
            dimension=config.get("dimension", DIMENSION),
        )
    return IndexConfig(model=DEFAULT_EMBED_MODEL, dimension=DIMENSION)


# ===========================================================================
# BATCH INGESTION CONFIGURATION
# ===========================================================================


@dataclass
class BatchIngestConfig:
    """Centralised configuration for batch ingestion."""

    pinecone_api_key: str
    openai_api_key: str
    redis_host: str
    redis_port: int
    redis_username: str
    redis_password: str

    local_path: str = "./data"  # Relative path
    index_name: str = "default"

    embed_model: str = "text-embedding-3-small"
    dimension: int = 1536

    chunk_tokens: int = 500
    chunk_overlap: int = 50
    embed_batch: int = 64
    upsert_batch: int = 200
    dedup_fetch_batch: int = INGEST_DEDUP_FETCH_BATCH_SIZE
    max_metadata_text_tokens: int = INGEST_METADATA_MAX_TEXT_TOKENS
    max_metadata_size: int = INGEST_METADATA_MAX_SIZE
    progress_log_interval: int = INGEST_PROGRESS_LOG_INTERVAL
    upsert_flush_seconds: float = INGEST_UPSERT_FLUSH_SECONDS
    upsert_join_timeout_seconds: float = INGEST_UPSERT_JOIN_TIMEOUT_SECONDS
    upsert_join_poll_seconds: float = INGEST_UPSERT_JOIN_POLL_SECONDS

    max_io_workers: int = 12
    max_parse_workers: int = 6
    upsert_workers: int = 2
    fra_supersession_single_threaded: bool = FRA_SUPERSESSION_SINGLE_THREADED
    max_pending_vectors: int = 2000
    openai_timeout: float = OPENAI_TIMEOUT_DEFAULT_S
    openai_connect_timeout: float = OPENAI_CONNECT_TIMEOUT_S
    openai_read_timeout: float = OPENAI_READ_TIMEOUT_S
    openai_write_timeout: float = OPENAI_WRITE_TIMEOUT_S
    openai_pool_timeout: float = OPENAI_POOL_TIMEOUT_S
    max_file_seconds: float = INGEST_MAX_FILE_SECONDS
    decode_responses: bool = True
    health_check_interval: int = 30
    upsert_strategy: str = "worker"

    max_file_mb: float = 10.0
    skip_existing: bool = True
    skip_successful_only: bool = True
    ext_whitelist: set[str] = field(
        default_factory=lambda: {"txt", "md", "csv", "json", "pdf", "docx"}
    )
    dry_run: bool = False

    cache_dir: str = ".cache/vector_ids"
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"

    export_events: bool = False
    export_events_file: str = "building_events.jsonl"
    prometheus_metrics_file: str = ""
    max_memory_mb: float = 512.0

    # -----------------------------------------------------------------------
    # ENVIRONMENT LOADERS
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "BatchIngestConfig":
        api_key = os.getenv("PINECONE_API_KEY")
        oai_key = os.getenv("OPENAI_API_KEY")
        redis_host = os.environ.get("REDIS_HOST")
        redis_port = int(os.environ.get("REDIS_PORT", 0)
                         ) if os.environ.get("REDIS_PORT") else 0
        redis_username = os.environ.get("REDIS_USERNAME", "")
        redis_password = os.environ.get("REDIS_PASSWORD", "")
        if not api_key:
            raise ConfigError("PINECONE_API_KEY not set")
        if not oai_key:
            raise ConfigError("OPENAI_API_KEY not set")
        if not redis_host or not redis_port:
            raise ConfigError("REDIS_HOST/REDIS_PORT not set")
        defaults = BatchIngestConfig(
            pinecone_api_key="", openai_api_key="", redis_host="", redis_port=0, redis_username="", redis_password="")

        return cls(
            pinecone_api_key=api_key,
            openai_api_key=oai_key,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_username=redis_username,
            redis_password=redis_password,
            local_path=os.getenv("LOCAL_PATH", defaults.local_path),
            index_name=os.getenv("INDEX_NAME", defaults.index_name),
            embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
            dimension=int(os.getenv("DIMENSION", str(defaults.dimension))),
            chunk_tokens=int(
                os.getenv("CHUNK_TOKENS", str(defaults.chunk_tokens))),
            chunk_overlap=int(
                os.getenv("CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            embed_batch=int(
                os.getenv("EMBED_BATCH", str(defaults.embed_batch))),
            upsert_batch=int(
                os.getenv("UPSERT_BATCH", str(defaults.upsert_batch))),
            dedup_fetch_batch=int(
                os.getenv("DEDUP_FETCH_BATCH", str(defaults.dedup_fetch_batch))),
            max_file_mb=float(
                os.getenv("MAX_FILE_MB", str(defaults.max_file_mb))),
            skip_existing=os.getenv("SKIP_EXISTING", "true").lower() == "true",
            skip_successful_only=os.getenv(
                "SKIP_SUCCESSFUL_ONLY",
                str(defaults.skip_successful_only),
            ).lower() in ("1", "true", "yes"),
            max_io_workers=int(
                os.getenv("MAX_IO_WORKERS", str(defaults.max_io_workers))),
            max_parse_workers=int(
                os.getenv("MAX_PARSE_WORKERS", str(defaults.max_parse_workers))),
            upsert_workers=int(
                os.getenv("UPSERT_WORKERS", str(defaults.upsert_workers))),
            max_pending_vectors=int(
                os.getenv("MAX_PENDING_VECTORS", str(defaults.max_pending_vectors))),
            openai_timeout=float(
                os.getenv("OPENAI_TIMEOUT", str(defaults.openai_timeout))),
            openai_connect_timeout=float(
                os.getenv("OPENAI_CONNECT_TIMEOUT",
                          str(defaults.openai_connect_timeout))),
            openai_read_timeout=float(
                os.getenv("OPENAI_READ_TIMEOUT",
                          str(defaults.openai_read_timeout))),
            openai_write_timeout=float(
                os.getenv("OPENAI_WRITE_TIMEOUT",
                          str(defaults.openai_write_timeout))),
            openai_pool_timeout=float(
                os.getenv("OPENAI_POOL_TIMEOUT",
                          str(defaults.openai_pool_timeout))),
            decode_responses=os.getenv(
                "DECODE_RESPONSES", "true").lower() == "true",
            health_check_interval=int(
                os.getenv("HEALTH_CHECK_INTERVAL", str(defaults.health_check_interval))),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
            upsert_strategy=os.getenv(
                "UPSERT_STRATEGY",
                defaults.upsert_strategy,
            ),
            export_events=os.getenv(
                "EXPORT_EVENTS", "false").lower() in ("1", "true", "yes"),
            export_events_file=os.getenv(
                "EXPORT_EVENTS_FILE", defaults.export_events_file),
            prometheus_metrics_file=os.getenv(
                "PROMETHEUS_METRICS_FILE", defaults.prometheus_metrics_file),
            dry_run=os.getenv("DRY_RUN", "false").lower() in (
                "1", "true", "yes"),
            max_metadata_text_tokens=int(
                os.getenv("MAX_METADATA_TEXT_TOKENS",
                          str(defaults.max_metadata_text_tokens))
            ),
            max_metadata_size=int(
                os.getenv("MAX_METADATA_SIZE",
                          str(defaults.max_metadata_size))
            ),
            progress_log_interval=int(
                os.getenv("PROGRESS_LOG_INTERVAL",
                          str(defaults.progress_log_interval))
            ),
            upsert_flush_seconds=float(
                os.getenv("UPSERT_FLUSH_SECONDS",
                          str(defaults.upsert_flush_seconds))
            ),
            upsert_join_timeout_seconds=float(
                os.getenv("UPSERT_JOIN_TIMEOUT_SECONDS",
                          str(defaults.upsert_join_timeout_seconds))
            ),
            upsert_join_poll_seconds=float(
                os.getenv("UPSERT_JOIN_POLL_SECONDS",
                          str(defaults.upsert_join_poll_seconds))
            ),
            fra_supersession_single_threaded=os.getenv(
                "FRA_SUPERSESSION_SINGLE_THREADED",
                str(defaults.fra_supersession_single_threaded),
            ).lower() in ("1", "true", "yes"),
            max_memory_mb=float(
                os.getenv("MAX_MEMORY_MB", str(defaults.max_memory_mb))),
            max_file_seconds=float(
                os.getenv("MAX_FILE_SECONDS", str(defaults.max_file_seconds))),
        )

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------

    def validate(self) -> None:

        if self.dimension not in [1536, 3072]:
            raise ConfigError(f"Invalid embedding dimension: {self.dimension}")

        if self.chunk_tokens < 100:
            raise ConfigError("chunk_tokens too small (<100)")

        if self.chunk_overlap >= self.chunk_tokens:
            raise ConfigError("chunk_overlap must be < chunk_tokens")

        if self.max_io_workers < 1:
            raise ConfigError("max_io_workers must be >= 1")

        if self.max_parse_workers < 1:
            raise ConfigError("max_parse_workers must be >= 1")
        if self.upsert_workers < 1:
            raise ConfigError("upsert_workers must be >= 1")
        if self.fra_supersession_single_threaded and self.upsert_workers != 1:
            logging.warning(
                "fra_supersession_single_threaded is enabled; forcing upsert_workers=1 "
                "(was %d) to avoid global lock contention.",
                self.upsert_workers,
            )
            self.upsert_workers = 1

        if self.embed_batch < 1 or self.embed_batch > 2048:
            raise ConfigError("embed_batch out of range (1-2048)")

        if self.upsert_batch < 1 or self.upsert_batch > 1000:
            raise ConfigError("upsert_batch out of range (1-1000)")

        if self.dedup_fetch_batch < 1 or self.dedup_fetch_batch > 1000:
            raise ConfigError("dedup_fetch_batch out of range (1-1000)")

        if self.max_pending_vectors < 1:
            raise ConfigError("max_pending_vectors must be >= 1")
        if self.openai_timeout <= 0:
            raise ConfigError("openai_timeout must be > 0")
        if self.openai_connect_timeout <= 0:
            raise ConfigError("openai_connect_timeout must be > 0")
        if self.openai_read_timeout <= 0:
            raise ConfigError("openai_read_timeout must be > 0")
        if self.openai_write_timeout <= 0:
            raise ConfigError("openai_write_timeout must be > 0")
        if self.openai_pool_timeout <= 0:
            raise ConfigError("openai_pool_timeout must be > 0")
        if self.max_file_seconds <= 0:
            raise ConfigError("max_file_seconds must be > 0")

        if self.max_metadata_text_tokens < 50 or self.max_metadata_text_tokens > 4096:
            raise ConfigError(
                "max_metadata_text_tokens out of range (50-4096)")
        if self.max_metadata_size < 1024:
            raise ConfigError("max_metadata_size too small (<1024)")
        if self.max_metadata_size > 262144:
            raise ConfigError("max_metadata_size too large (>262144)")
        if self.progress_log_interval < 1:
            raise ConfigError("progress_log_interval must be >= 1")
        if self.upsert_strategy not in ("worker", "inline"):
            raise ConfigError(
                f"Invalid upsert_strategy: {self.upsert_strategy}")
        if self.upsert_flush_seconds < 0:
            raise ConfigError("upsert_flush_seconds must be >= 0")
        if self.upsert_join_timeout_seconds <= 0:
            raise ConfigError("upsert_join_timeout_seconds must be > 0")
        if self.upsert_join_poll_seconds <= 0:
            raise ConfigError("upsert_join_poll_seconds must be > 0")
        if self.openai_timeout > self.max_file_seconds:
            logging.warning(
                "openai_timeout (%.1fs) exceeds max_file_seconds (%.1fs); "
                "per-file timeout may still abort long requests.",
                self.openai_timeout,
                self.max_file_seconds,
            )

        # Soft memory estimate for pending vectors.
        # Embeddings are float32 => 4 bytes per dimension. We apply a small overhead factor
        # to account for Python/container overhead and queues.
        overhead_factor = 1.75
        vector_bytes = self.dimension * 4  # float32
        estimated_mb = (self.max_pending_vectors *
                        vector_bytes * overhead_factor) / (1024 * 1024)

        if estimated_mb > self.max_memory_mb:
            logging.warning(
                "max_pending_vectors (%s) may use ~%.1fMB (incl. overhead), exceeds limit %.1fMB. "
                "Consider lowering MAX_PENDING_VECTORS, dimension, or batch sizes.",
                self.max_pending_vectors,
                estimated_mb,
                self.max_memory_mb,
            )

        path = Path(self.local_path)
        if not path.exists():
            raise ConfigError(f"local_path does not exist: {self.local_path}")
        if not path.is_dir():
            raise ConfigError(
                f"local_path is not a directory: {self.local_path}")
        if getattr(self, "dry_run", False):
            # Dry-run must not depend on Pinecone state (listing/upserting)
            self.skip_existing = False
        if not getattr(self, "dry_run", False):
            if not self.redis_host or not self.redis_port:
                raise ConfigError(
                    "redis_host/redis_port must be set in non-dry-run")
            if self.redis_username and not self.redis_password:
                raise ConfigError(
                    "redis_password must be set when redis_username is provided"
                )
            logging.info(
                "Redis target: %s:%s",
                self.redis_host,
                self.redis_port,
            )

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ANSWER_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_NAMESPACE",
    "DIMENSION",
    "DocumentTypes",
    "FRA_RISK_ITEMS_NAMESPACE",
    "INDEX_CONFIGS",
    "MIN_SCORE_THRESHOLD",
    "NAMESPACE_MAPPINGS",
    "RISK_LEVEL_MAP",
    "SEARCH_ALL_NAMESPACES",
    "TARGET_INDEXES",
    "get_display_namespace",
    "get_internal_namespace",
    "normalise_ns",
    "resolve_namespace",
    "_route_namespace",
    "IndexConfig",
    "get_index_config",
    "BatchIngestConfig",
]
