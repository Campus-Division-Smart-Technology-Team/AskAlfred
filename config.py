#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for AskAlfred chatbot.
Enhanced with validation and type safety.
"""

from typing import TypedDict, Optional, Set, Dict
from dataclasses import dataclass, field
import os
import logging
from pathlib import Path

import streamlit as st


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

# ---------------------------------------------------------------------------
# CORE CONSTANTS
# ---------------------------------------------------------------------------
# Debug logging for query routing
DEBUG_QUERY_MODE = True

DEFAULT_NAMESPACE: Optional[str] = None  # None means default namespace

# ---------------------------------------------------------------------------
# QUERY MANAGER CONFIGURATION
# ---------------------------------------------------------------------------
# Query Manager Feature Flag
USE_QUERY_MANAGER = True  # Set to True to enable new system
# Query Manager Configuration
QUERY_MANAGER_CONFIG = {
    'enable_caching': False,  # Enable query result caching
    'cache_ttl_seconds': 300,  # Cache TTL
    'enable_metrics': True,    # Track query metrics
    'log_level': 'INFO'        # Logging level
}

# ---------------------------------------------------------------------------
# QUERY HANDLER CONFIGURATION
# ---------------------------------------------------------------------------
# Centralised configuration for query handlers
QUERY_HANDLER_CONFIG = {
    'conversational': {
        'priority': 1,
        'enabled': True
    },
    'maintenance': {
        'priority': 2,
        'enabled': True,
        'min_confidence': 0.7
    },
    'ranking': {
        'priority': 3,
        'enabled': True
    },
    'property': {
        'priority': 4,
        'enabled': True
    },
    'counting': {
        'priority': 5,
        'enabled': True
    },
    'semantic_search': {
        'priority': 99,
        'enabled': True,
        'min_score_threshold': 0.7
    }
}

# ---------------------------------------------------------------------------
# NAMESPACE UTILITIES
# ---------------------------------------------------------------------------


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


# ===========================================================================
# NAMESPACE MAPPINGS
# ===========================================================================
# Maps internal document types → Pinecone namespaces
# If a doc_type is not listed, ingestion should fail or fall back intentionally.
NAMESPACE_MAPPINGS: Dict[str, str] = {
    "planon_data": "planon_data",
    "maintenance_request": "maintenance_requests",
    "maintenance_job": "maintenance_jobs",
    "fire_risk_assessment": "fire_risk_assessments",
    "operational_doc": "operational_docs",
    "unknown": "operational_docs"
}


def resolve_namespace(doc_type: Optional[str]) -> Optional[str]:
    if not doc_type:
        logging.debug(
            "resolve_namespace: no doc_type supplied, using default namespace.")
        return DEFAULT_NAMESPACE
    try:
        return NAMESPACE_MAPPINGS[doc_type]
    except KeyError:
        raise ValueError(f"Undefined doc_type: {doc_type}")


TARGET_INDEXES = ["local-docs"]
SEARCH_ALL_NAMESPACES = True
DEFAULT_EMBED_MODEL = os.getenv(
    "DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
DIMENSION = 1536
MIN_SCORE_THRESHOLD = 0.3
INDEX_CONFIGS = {
    "local-docs": {"model": DEFAULT_EMBED_MODEL, "dimension": DIMENSION}
}


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


class DocumentTypes:
    MAINTENANCE_REQUEST = "maintenance_request"
    MAINTENANCE_JOB = "maintenance_job"
    FIRE_RISK_ASSESSMENT = "fire_risk_assessment"
    OPERATIONAL_DOC = "operational_doc"
    PLANON_DATA = "planon_data"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# BATCH INGESTION CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class BatchIngestConfig:
    """Centralised configuration for batch ingestion."""

    pinecone_api_key: str
    openai_api_key: str

    # Fixed: Use relative path instead of hardcoded Windows path
    local_path: str = "./Alfred"
    index_name: str = "local-docs"

    embed_model: str = "text-embedding-3-small"
    dimension: int = 1536

    chunk_tokens: int = 500
    chunk_overlap: int = 50
    embed_batch: int = 64
    upsert_batch: int = 64

    max_workers: int = 4  # Fixed: Increased from 1 to 4 for better performance
    max_pending_vectors: int = 10000
    openai_timeout: float = 30.0

    max_file_mb: float = 10.0  # Fixed: Added reasonable default limit (10MB)
    skip_existing: bool = True
    ext_whitelist: Set[str] = field(
        default_factory=lambda: {"txt", "md", "csv", "json", "pdf", "docx"}
    )
    dry_run: bool = False

    cache_dir: str = ".cache/vector_ids"
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"

    export_events: bool = False
    export_events_file: str = "building_events.jsonl"
    MAX_METADATA_TEXT_TOKENS = 800

    # -----------------------------------------------------------------------
    # ENVIRONMENT LOADERS
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "BatchIngestConfig":
        api_key = os.getenv("PINECONE_API_KEY")
        oai_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        if not oai_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        defaults = BatchIngestConfig(pinecone_api_key="", openai_api_key="")

        return cls(
            pinecone_api_key=api_key,
            openai_api_key=oai_key,
            local_path=os.getenv("LOCAL_PATH", defaults.local_path),
            index_name=os.getenv("INDEX_NAME", defaults.index_name),
            embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
            dimension=int(os.getenv("DIMENSION", defaults.dimension)),
            chunk_tokens=int(os.getenv("CHUNK_TOKENS", defaults.chunk_tokens)),
            chunk_overlap=int(
                os.getenv("CHUNK_OVERLAP", defaults.chunk_overlap)),
            embed_batch=int(os.getenv("EMBED_BATCH", defaults.embed_batch)),
            upsert_batch=int(os.getenv("UPSERT_BATCH", defaults.upsert_batch)),
            max_file_mb=float(os.getenv("MAX_FILE_MB", defaults.max_file_mb)),
            skip_existing=os.getenv("SKIP_EXISTING", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", defaults.max_workers)),
            max_pending_vectors=int(
                os.getenv("MAX_PENDING_VECTORS", defaults.max_pending_vectors)),
            openai_timeout=float(
                os.getenv("OPENAI_TIMEOUT", defaults.openai_timeout)),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level),
            export_events=os.getenv(
                "EXPORT_EVENTS", "false").lower() in ("1", "true", "yes"),
            export_events_file=os.getenv(
                "EXPORT_EVENTS_FILE", defaults.export_events_file),
            dry_run=os.getenv("DRY_RUN", "false").lower() in (
                "1", "true", "yes"),
        )

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------

    def validate(self) -> None:

        if self.dimension not in [1536, 3072]:
            raise ValueError(f"Invalid embedding dimension: {self.dimension}")

        if self.chunk_tokens < 100:
            raise ValueError("chunk_tokens too small (<100)")

        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be < chunk_tokens")

        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if self.embed_batch < 1 or self.embed_batch > 2048:
            raise ValueError("embed_batch out of range (1-2048)")

        if self.upsert_batch < 1 or self.upsert_batch > 1000:
            raise ValueError("upsert_batch out of range (1-1000)")

        path = Path(self.local_path)
        if not path.exists():
            raise ValueError(f"local_path does not exist: {self.local_path}")
        if not path.is_dir():
            raise ValueError(
                f"local_path is not a directory: {self.local_path}")
        if getattr(self, "dry_run", False):
            # Dry-run must not depend on Pinecone state (listing/upserting)
            self.skip_existing = False

        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
