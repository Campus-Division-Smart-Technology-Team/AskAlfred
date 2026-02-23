#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants for AskAlfred configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ===========================================================================
# CORE CONSTANTS
# ===========================================================================
# Debug logging for query routing
DEBUG_QUERY_MODE = True

DEFAULT_NAMESPACE: Optional[str] = None  # None means default namespace

# ===========================================================================
# BUILDING RESOLUTION & ORDER CONFIGURATION
# ===========================================================================
BUILDING_FUZZY_STRONG = 0.75
BUILDING_FUZZY_WEAK = 0.50
BUILDING_TEXT_CUTOFF = 0.80
BUILDING_TEXT_CONFIDENCE = 0.60
BUILDING_REVIEW_TEXT_MIN_CONFIDENCE = 0.70
BUILDING_REVIEW_FILENAME_MIN_CONFIDENCE = 0.75
BUILDING_TEXT_FIRST_LINES = 50
BUILDING_TEXT_NGRAM_MAX = 5
BUILDING_TEXT_NGRAM_MIN = 2
BUILDING_NAME_FIELDS = [
    "canonical_building_name",
    "building_name",
    "Property names",
    "UsrFRACondensedPropertyName",
]

# ===========================================================================
# INGEST / RETRY CONFIGURATION
# ===========================================================================
INGEST_LOW_CONFIDENCE_WARN = 0.70
INGEST_FETCH_MAX_SIZE_MB = 100
INGEST_BACKOFF_BASE = 0.5
INGEST_BACKOFF_CAP = 10.0
INGEST_BACKOFF_JITTER_MIN = 0.5
INGEST_BACKOFF_JITTER_SPAN = 1.0
INGEST_EMBED_MAX_RETRIES = 2
INGEST_RETRY_ATTEMPTS = 3
INGEST_RETRY_EXP_MULTIPLIER = 1
INGEST_RETRY_EXP_MIN = 2
INGEST_RETRY_EXP_MAX = 10
INGEST_EMBED_BATCH_SIZE = 32
INGEST_VERIFY_FETCH_BATCH_SIZE = 100
INGEST_DEDUP_FETCH_BATCH_SIZE = 100
INGEST_VERIFY_BACKOFF_BASE = 0.2
INGEST_VERIFY_BACKOFF_CAP = 1.5
INGEST_UPSERT_SPLIT_MAX_DEPTH = 4
INGEST_UPSERT_SPLIT_MIN_BATCH_SIZE = 10
INGEST_METADATA_MAX_SIZE = 10240  # Previously 40960
INGEST_METADATA_CACHE_SIZE = 2048
INGEST_METADATA_MAX_TEXT_TOKENS = 250
INGEST_VECTOR_BUFFER_MAX_SIZE = 10000
INGEST_PROGRESS_LOG_INTERVAL = 10
INGEST_PROCESSING_LEASE_SECONDS = 10 * 60  # Previously 900
INGEST_UPSERT_FLUSH_SECONDS = 2.0
INGEST_UPSERT_JOIN_TIMEOUT_SECONDS = 900.0
INGEST_UPSERT_JOIN_POLL_SECONDS = 2.0
INGEST_FILE_TTL_SUCCESS_SECONDS = 30 * 60  # Previously 30 * 24 * 60 * 60
INGEST_FILE_TTL_FAILED_SECONDS = 10 * 60  # Previously 7 * 24 * 60 * 60
INGEST_FILE_TTL_PROCESSING_SECONDS = 10 * 60  # Previously 24 * 60 * 60
INGEST_JOB_TTL_DEFAULT_SECONDS = 15 * 60  # Previously 7 * 24 * 60 * 60
INGEST_JOB_TTL_SUPERSEDE_SECONDS = 15 * 60  # Previously 30 * 24 * 60 * 60
OPENAI_TIMEOUT_DEFAULT_S = 60.0
OPENAI_CONNECT_TIMEOUT_S = 10.0
OPENAI_READ_TIMEOUT_S = 60.0
OPENAI_WRITE_TIMEOUT_S = 60.0
OPENAI_POOL_TIMEOUT_S = 30.0
INGEST_MAX_FILE_SECONDS = 900.0

# ===========================================================================
# PINECONE CONFIGURATION
# ===========================================================================
PAGE_LIMIT = 1000

# ===========================================================================
# QUERY ROUTING CONFIGURATION
# ===========================================================================
QUERY_RULE_OVERRIDE_THRESHOLD = 0.75
QUERY_CONF_THRESHOLD = 0.60
QUERY_FOLLOWUP_ML_CONF_THRESHOLD = 0.55
QUERY_FOLLOWUP_MAX_TOKENS = 2
QUERY_MIN_LENGTH = 2
QUERY_MAX_LENGTH = 1000

# ===========================================================================
# INTENT CLASSIFIER CONFIGURATION
# ===========================================================================
INTENT_MEAN_SIMILARITY_WEIGHT = 0.7
INTENT_MAX_EXAMPLE_SIMILARITY_WEIGHT = 0.3
INTENT_SOFTMAX_TEMPERATURE = 0.2
INTENT_FOLLOWUP_BOOST_FACTOR = 0.03
INTENT_BUILDING_MAINTENANCE_BIAS = 0.05
INTENT_BUILDING_COUNTING_BIAS = 0.05
INTENT_BUILDING_CONDITION_BIAS = 0.05
INTENT_BUILDING_SEMANTIC_BIAS = 0.02
INTENT_BUSINESS_COUNTING_BIAS = 0.02
INTENT_BUSINESS_SEMANTIC_BIAS = 0.02
INTENT_BUSINESS_CONDITION_BIAS = 0.02
INTENT_HIGHER_UPPER_CONFIDENCE = 0.85
INTENT_LOWER_UPPER_CONFIDENCE = 0.80
INTENT_HIGHER_MIDDLE_CONFIDENCE = 0.75
INTENT_LOWER_MIDDLE_CONFIDENCE = 0.70
INTENT_LOWEST_CONFIDENCE = 0.60
INTENT_CONFIDENCE_THRESHOLD = 0.65
INTENT_QUERY_CACHE_MAX_SIZE = 32

# ===========================================================================
# FRA LOCKING CONFIGURATION
# ===========================================================================
FRA_LOCK_TIMEOUT_SECONDS = 10.0
FRA_LOCK_ACQUIRE_SECONDS = 30.0
FRA_PARTITION_KEY_BUCKET_SIZE = 64
FRA_SUPERSESSION_SINGLE_THREADED = False
REDIS_LOCK_KEY_PREFIX = "fra:supersede"
REDIS_LOCK_DEFAULT_TTL_MS = 60_000
REDIS_LOCK_RETRY_INTERVAL_S = 0.2
REDIS_LOCK_JITTER_S = 0.1
REDIS_POOL_MAX_CONNECTIONS = 20
REDIS_POOL_SOCKET_TIMEOUT_S = 5.0
REDIS_POOL_SOCKET_CONNECT_TIMEOUT_S = 5.0
REDIS_POOL_HEALTH_CHECK_INTERVAL_S = 30.0

# ===========================================================================
# FRA TRIAGE CONFIGURATION
# ===========================================================================
FRA_LONG_OVERDUE_DAYS = 90
FRA_CRITICAL_OVERDUE_DAYS = 180
FRA_EXTREME_OVERDUE_DAYS = 365
FRA_MAX_DAYS_SANITY = 3650
FRA_RISK_BASE_SCORES = {1: 10, 2: 20, 3: 40, 4: 70, 5: 100}
FRA_OVERDUE_DIVISOR_DAYS = 365
FRA_OVERDUE_MULTIPLIER_CAP = 0.5
NO_JOB_REF_SCORE_MULTIPLIER = 0.9
FRA_NO_JOB_REF_MIN_RISK_LEVEL = 3
FRA_RISK_SCORE_MAX = 100
FRA_PRIORITY_HIGH_RISK_LEVEL = 4
FRA_PRIORITY_MEDIUM_RISK_LEVEL = 3

# ===========================================================================
# FRA PARSER CONFIGURATION
# ===========================================================================
FRA_PARSING_FIELD_WEIGHTS = {
    "issue_description": 0.3,
    "proposed_solution": 0.2,
    "person_responsible": 0.2,
    "job_reference": 0.15,
    "expected_completion_date": 0.15,
}
SPLIT_RISK_FIXES = [
    (r"\bTolera\s*\n\s*ble\b", "Tolerable"),
    (r"\bModera\s*\n\s*te\b", "Moderate"),
    (r"\bSubstan\s*\n\s*tial\b", "Substantial"),
    (r"\bIntolera\s*\n\s*ble\b", "Intolerable"),
    (r"\bTrivi\s*\n\s*al\b", "Trivial"),
]
FRA_WARNING_PENALTY_STEP = 0.05
FRA_WARNING_PENALTY_CAP = 0.2

# ===========================================================================
# MODEL PATH CONFIGURATION
# ===========================================================================
LOCAL_MODEL_DIR = Path("models/all-MiniLM-L6-v2")

# ===========================================================================
# SESSION MANAGER CONFIGURATION
# ===========================================================================
SESSION_LONG_MESSAGE_LENGTH = 200
SESSION_HIGH_CONFIDENCE_THRESHOLD = 0.8
SESSION_SUMMARY_PREVIEW_LEN = 100
SESSION_SUMMARY_MAX_KEY_POINTS = 15
SESSION_IMPORTANT_KEEP_RATIO = 5
SESSION_IMPORTANT_MIN_KEEP = 5
SESSION_IMPORTANCE_BASELINE = 0.5
SESSION_IMPORTANCE_USER_BONUS = 0.1
SESSION_IMPORTANCE_INTENT_BONUS = 0.2
SESSION_IMPORTANCE_BUILDING_BONUS = 0.15
SESSION_IMPORTANCE_CONFIDENCE_BONUS = 0.1
SESSION_IMPORTANCE_ERROR_BONUS = 0.2
SESSION_IMPORTANCE_LONG_MESSAGE_BONUS = 0.05

# ===========================================================================
# UI CONFIGURATION
# ===========================================================================
UI_TOP_K_MIN = 1
UI_TOP_K_MAX = 25
UI_TOP_K_DEFAULT = 5
UI_SNIPPET_MAX_CHARS = 500
UI_SUMMARY_MAX_TOKENS = 150
UI_RECENT_TURNS_FOR_SUMMARY = 4

# ===========================================================================
# ANSWER GENERATION CONFIGURATION
# ===========================================================================
ANSWER_TEMPERATURE_DEFAULT = 0.2
ANSWER_TEMPERATURE_COMPARE = 0.3

# ===========================================================================
# BUILDING UTILITIES CONFIGURATION
# ===========================================================================
BUILDING_UTILS_FUZZY_MATCH_THRESHOLD = 0.80
BUILDING_UTILS_MIN_NAME_LENGTH = 3

# ===========================================================================
# QUERY MANAGER CONFIGURATION
# ===========================================================================
# Query Manager Feature Flag
USE_QUERY_MANAGER = True  # Set to True to enable new system
# Query Manager Configuration
QUERY_MANAGER_CONFIG = {
    "enable_caching": False,  # Enable query result caching
    "cache_ttl_seconds": 300,  # Cache TTL
    "enable_metrics": True,    # Track query metrics
    "log_level": "INFO",       # Logging level
}

# ===========================================================================
# QUERY HANDLER CONFIGURATION
# ===========================================================================
# Centralised configuration for query handlers
QUERY_HANDLER_CONFIG = {
    "conversational": {
        "priority": 1,
        "enabled": True,
    },
    "maintenance": {
        "priority": 2,
        "enabled": True,
        "min_confidence": 0.7,
    },
    "ranking": {
        "priority": 3,
        "enabled": True,
    },
    "property": {
        "priority": 4,
        "enabled": True,
    },
    "counting": {
        "priority": 5,
        "enabled": True,
    },
    "semantic_search": {
        "priority": 99,
        "enabled": True,
        "min_score_threshold": 0.7,
    },
}

# ===========================================================================
# NAMESPACE MAPPINGS
# ===========================================================================
# Maps internal document types -> Pinecone namespaces
# If a doc_type is not listed, ingestion should fail or fall back intentionally.
NAMESPACE_MAPPINGS: dict[str, str] = {
    "planon_data": "planon_data",
    "maintenance_request": "maintenance_requests",
    "maintenance_job": "maintenance_jobs",
    "fire_risk_assessment": "fire_risk_assessments",
    "operational_doc": "operational_docs",
    "fra_risk_item": "fra_risk_items",
    "unknown": "operational_docs",
}


class DocumentTypes:
    MAINTENANCE_REQUEST = "maintenance_request"
    MAINTENANCE_JOB = "maintenance_job"
    FIRE_RISK_ASSESSMENT = "fire_risk_assessment"
    FRA_RISK_ITEM = "fra_risk_item"
    OPERATIONAL_DOC = "operational_doc"
    PLANON_DATA = "planon_data"
    UNKNOWN = "unknown"


FRA_RISK_ITEMS_NAMESPACE = NAMESPACE_MAPPINGS[DocumentTypes.FRA_RISK_ITEM]

TARGET_INDEXES = ["local-docs"]
SEARCH_ALL_NAMESPACES = True
DEFAULT_EMBED_MODEL = os.getenv(
    "DEFAULT_EMBED_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
DIMENSION = 1536
MIN_SCORE_THRESHOLD = 0.3
INDEX_CONFIGS = {
    "local-docs": {"model": DEFAULT_EMBED_MODEL, "dimension": DIMENSION},
}

# ===========================================================================
# RISK LEVEL MAPPINGS
# ===========================================================================
RISK_LEVEL_MAP: dict[str, str] = {
    "1": "Trivial",
    "2": "Tolerable",
    "3": "Moderate",
    "4": "Substantial",
    "5": "Intolerable",
}

__all__ = [
    "DEBUG_QUERY_MODE",
    "DEFAULT_NAMESPACE",
    "BUILDING_FUZZY_STRONG",
    "BUILDING_FUZZY_WEAK",
    "BUILDING_TEXT_CUTOFF",
    "BUILDING_TEXT_CONFIDENCE",
    "BUILDING_REVIEW_TEXT_MIN_CONFIDENCE",
    "BUILDING_REVIEW_FILENAME_MIN_CONFIDENCE",
    "BUILDING_TEXT_FIRST_LINES",
    "BUILDING_TEXT_NGRAM_MAX",
    "BUILDING_TEXT_NGRAM_MIN",
    "BUILDING_NAME_FIELDS",
    "INGEST_LOW_CONFIDENCE_WARN",
    "INGEST_FETCH_MAX_SIZE_MB",
    "INGEST_BACKOFF_BASE",
    "INGEST_BACKOFF_CAP",
    "INGEST_BACKOFF_JITTER_MIN",
    "INGEST_BACKOFF_JITTER_SPAN",
    "INGEST_EMBED_MAX_RETRIES",
    "INGEST_RETRY_ATTEMPTS",
    "INGEST_RETRY_EXP_MULTIPLIER",
    "INGEST_RETRY_EXP_MIN",
    "INGEST_RETRY_EXP_MAX",
    "INGEST_EMBED_BATCH_SIZE",
    "INGEST_VERIFY_FETCH_BATCH_SIZE",
    "INGEST_DEDUP_FETCH_BATCH_SIZE",
    "INGEST_VERIFY_BACKOFF_BASE",
    "INGEST_VERIFY_BACKOFF_CAP",
    "INGEST_UPSERT_SPLIT_MAX_DEPTH",
    "INGEST_UPSERT_SPLIT_MIN_BATCH_SIZE",
    "INGEST_METADATA_MAX_SIZE",
    "INGEST_METADATA_MAX_TEXT_TOKENS",
    "INGEST_VECTOR_BUFFER_MAX_SIZE",
    "INGEST_PROGRESS_LOG_INTERVAL",
    "INGEST_PROCESSING_LEASE_SECONDS",
    "INGEST_UPSERT_FLUSH_SECONDS",
    "INGEST_UPSERT_JOIN_TIMEOUT_SECONDS",
    "INGEST_UPSERT_JOIN_POLL_SECONDS",
    "INGEST_FILE_TTL_SUCCESS_SECONDS",
    "INGEST_FILE_TTL_FAILED_SECONDS",
    "INGEST_FILE_TTL_PROCESSING_SECONDS",
    "INGEST_JOB_TTL_DEFAULT_SECONDS",
    "INGEST_JOB_TTL_SUPERSEDE_SECONDS",
    "OPENAI_TIMEOUT_DEFAULT_S",
    "OPENAI_CONNECT_TIMEOUT_S",
    "OPENAI_READ_TIMEOUT_S",
    "OPENAI_WRITE_TIMEOUT_S",
    "OPENAI_POOL_TIMEOUT_S",
    "INGEST_MAX_FILE_SECONDS",
    "PAGE_LIMIT",
    "QUERY_RULE_OVERRIDE_THRESHOLD",
    "QUERY_CONF_THRESHOLD",
    "QUERY_FOLLOWUP_ML_CONF_THRESHOLD",
    "QUERY_FOLLOWUP_MAX_TOKENS",
    "QUERY_MIN_LENGTH",
    "QUERY_MAX_LENGTH",
    "INTENT_MEAN_SIMILARITY_WEIGHT",
    "INTENT_MAX_EXAMPLE_SIMILARITY_WEIGHT",
    "INTENT_SOFTMAX_TEMPERATURE",
    "INTENT_FOLLOWUP_BOOST_FACTOR",
    "INTENT_BUILDING_MAINTENANCE_BIAS",
    "INTENT_BUILDING_COUNTING_BIAS",
    "INTENT_BUILDING_CONDITION_BIAS",
    "INTENT_BUILDING_SEMANTIC_BIAS",
    "INTENT_BUSINESS_COUNTING_BIAS",
    "INTENT_BUSINESS_SEMANTIC_BIAS",
    "INTENT_BUSINESS_CONDITION_BIAS",
    "INTENT_HIGHER_UPPER_CONFIDENCE",
    "INTENT_LOWER_UPPER_CONFIDENCE",
    "INTENT_HIGHER_MIDDLE_CONFIDENCE",
    "INTENT_LOWER_MIDDLE_CONFIDENCE",
    "INTENT_LOWEST_CONFIDENCE",
    "INTENT_CONFIDENCE_THRESHOLD",
    "INTENT_QUERY_CACHE_MAX_SIZE",
    "FRA_LOCK_TIMEOUT_SECONDS",
    "FRA_LOCK_ACQUIRE_SECONDS",
    "FRA_SUPERSESSION_SINGLE_THREADED",
    "FRA_PARTITION_KEY_BUCKET_SIZE",
    "FRA_LONG_OVERDUE_DAYS",
    "FRA_CRITICAL_OVERDUE_DAYS",
    "FRA_EXTREME_OVERDUE_DAYS",
    "FRA_MAX_DAYS_SANITY",
    "FRA_RISK_BASE_SCORES",
    "FRA_OVERDUE_DIVISOR_DAYS",
    "FRA_OVERDUE_MULTIPLIER_CAP",
    "NO_JOB_REF_SCORE_MULTIPLIER",
    "FRA_NO_JOB_REF_MIN_RISK_LEVEL",
    "FRA_RISK_SCORE_MAX",
    "FRA_PRIORITY_HIGH_RISK_LEVEL",
    "FRA_PRIORITY_MEDIUM_RISK_LEVEL",
    "FRA_PARSING_FIELD_WEIGHTS",
    "FRA_WARNING_PENALTY_STEP",
    "FRA_WARNING_PENALTY_CAP",
    "REDIS_LOCK_KEY_PREFIX",
    "REDIS_LOCK_DEFAULT_TTL_MS",
    "REDIS_LOCK_RETRY_INTERVAL_S",
    "REDIS_LOCK_JITTER_S",
    "REDIS_POOL_MAX_CONNECTIONS",
    "REDIS_POOL_SOCKET_TIMEOUT_S",
    "REDIS_POOL_SOCKET_CONNECT_TIMEOUT_S",
    "REDIS_POOL_HEALTH_CHECK_INTERVAL_S",
    "LOCAL_MODEL_DIR",
    "SESSION_LONG_MESSAGE_LENGTH",
    "SESSION_HIGH_CONFIDENCE_THRESHOLD",
    "SESSION_SUMMARY_PREVIEW_LEN",
    "SESSION_SUMMARY_MAX_KEY_POINTS",
    "SESSION_IMPORTANT_KEEP_RATIO",
    "SESSION_IMPORTANT_MIN_KEEP",
    "SESSION_IMPORTANCE_BASELINE",
    "SESSION_IMPORTANCE_USER_BONUS",
    "SESSION_IMPORTANCE_INTENT_BONUS",
    "SESSION_IMPORTANCE_BUILDING_BONUS",
    "SESSION_IMPORTANCE_CONFIDENCE_BONUS",
    "SESSION_IMPORTANCE_ERROR_BONUS",
    "SESSION_IMPORTANCE_LONG_MESSAGE_BONUS",
    "UI_TOP_K_MIN",
    "UI_TOP_K_MAX",
    "UI_TOP_K_DEFAULT",
    "UI_SNIPPET_MAX_CHARS",
    "UI_SUMMARY_MAX_TOKENS",
    "UI_RECENT_TURNS_FOR_SUMMARY",
    "ANSWER_TEMPERATURE_DEFAULT",
    "ANSWER_TEMPERATURE_COMPARE",
    "BUILDING_UTILS_FUZZY_MATCH_THRESHOLD",
    "BUILDING_UTILS_MIN_NAME_LENGTH",
    "USE_QUERY_MANAGER",
    "QUERY_MANAGER_CONFIG",
    "QUERY_HANDLER_CONFIG",
    "NAMESPACE_MAPPINGS",
    "DocumentTypes",
    "FRA_RISK_ITEMS_NAMESPACE",
    "TARGET_INDEXES",
    "SEARCH_ALL_NAMESPACES",
    "DEFAULT_EMBED_MODEL",
    "ANSWER_MODEL",
    "DIMENSION",
    "MIN_SCORE_THRESHOLD",
    "INDEX_CONFIGS",
    "RISK_LEVEL_MAP",
]
