# 🦍 Alfred V3 — Modular, Hybrid-Intent Building-Aware Search Assistant

Alfred is an intelligent, Streamlit-based search assistant for the University of Bristol's Campus Division.

It provides **multi-domain, building-aware search** across:
- Building Management Systems (BMS)
- Fire Risk Assessments and Fire Risk Action Items (FRAs)
- Planon property data (conditions, areas, metadata)
- Maintenance requests and job records  
- General RAG / semantic search across documentation

Powered by:
✅ OpenAI embeddings  
✅ Pinecone vector search  
✅ A hybrid **rule-based + ML intent classifier** pipeline  

---

## 🧠 Features 

Alfred V3 is a modular, transactional, and concurrency-safe ingestion pipeline with:

- Secure local file ingestion
- FRA action plan parsing with structured extraction
- Deterministic triage scoring & prioritisation
- Atomic supersession handling for FRA updates
- Vector storage via Pinecone
- Embedding via OpenAI
- Redis-backed job registry, locks & file registry
- Thread-safe stats, cache & vector buffering
- Dry-run mode for safe validation
- Metrics instrumentation
- Parallel IO + parsing workers
- Batched and retry-aware upsert coordination

Alfred V3 uses a **Hybrid Intent Routing System** with the `NLPIntentClassifier`:

### ✅ `intent_classifier.py` - NLPIntentClassifier
A sophisticated, context-aware intent classifier using a **quantized all-MiniLM-L6-v2 CT2 encoder** (`michaelfeil/ct2fast-all-MiniLM-L6-v2`) that:

**Core Features:**
- Loads pre-trained model from local `models/all-MiniLM-L6-v2/` directory or auto-downloads from Hugging Face
- Auto-extracts zipped models at startup for convenience
- Generates and caches intent embeddings for all query types (saved as JSON + NPZ in `intent_embeddings_cache.json` and `intent_embeddings_cache.npz`)
- Returns calibrated confidence scores using **softmax normalisation**
- Provides both semantic and pattern-based classification with automatic fallback

**Advanced Capabilities:**
- **Context-aware biasing**: Adjusts confidence scores based on `QueryContext` (detected buildings, business terms)
- **Hybrid classification**: Combines semantic similarity (70% mean + 30% max example) with pattern matching
- **Confidence threshold**: Default 0.65 threshold triggers pattern fallback for low-confidence predictions
- **Graceful degradation**: Falls back to pattern-only mode if CT2 encoder unavailable

**Intent Training Examples:**
The classifier is trained on domain-specific examples across 6 query types:
- `CONVERSATIONAL` (greetings, help requests)
- `MAINTENANCE` (PPM, jobs, requests)
- `RANKING` (largest, top N, comparisons)
- `PROPERTY_CONDITION` (derelict, condition A-D)
- `COUNTING` (how many, count)
- `SEMANTIC_SEARCH` (BMS config, FRA process, HVAC systems)

### ✅ Classification Behavior:
- If semantic confidence ≥ 0.65 → Uses semantic classification with context biasing
- If semantic confidence < 0.65 → Falls back to pattern-based classification
- Context biasing adjusts scores by up to 5% based on detected buildings and business terms
- If a handler declines during negotiation, QueryManager escalates automatically

---

## 🧠 Core Architecture Overview

Alfred's architecture follows a **modular, layered design**:

```
            ┌────────────────────────┐
            │      Streamlit UI      │
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │     QueryManager       │
            │  Hybrid Intent Router  │
            └──────────┬─────────────┘
                       │
       ┌───────────────┼────────────────────┐
       │ Rule Layer → Regex/Keyword Matching│
       │ ML Layer → NLPIntentClassifier     │
       └───────────────┬────────────────────┘
                       │
    ┌──────────────────▼────────────────────┐
    │         Handlers Layer                │
    │ (Conversational / Property /          │
    │  Maintenance / Counting / Ranking /   │
    │  SemanticSearch)                      │
    └───────────────────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │   search_core package  │
            └────────────────────────┘
```

---
## 🧠 Ingestion Architecture Overview

Alfred's ingestion architecture is mainly determined by **document-type** and the upsert strategy:

```
            ┌────────────────────────┐
            │     Local Files        │
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │ list_local_files_secure│
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │   Building Resolution  │
            │(Property CSV + aliases)│
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │   DocumentProcessor    │
            │  (extract + chunk +    │
            │   FRA vector path)     │
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │ FileIngestOrchestrator │
            │    (IO/parse pools)    │
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │ VectorWriteCoordinator │
            │ (batch + flush policy) │
            └──────────┬─────────────┘
                       │
            ┌──────────▼─────────────┐
            │ Upsert:worker or inline│
            └──────────┬─────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │     Pinecone Index     │
            └────────────────────────┘
```

Files are processed by `DocumentProcessor`, which extracts text, chunks, and (for FRA candidates) routes through the FRA vector extraction path before returning vectors. `VectorWriteCoordinator` then batches and flushes vectors either **inline** (direct upsert in the main thread) or via **worker** threads (queue + `_upsert_worker`) depending on `upsert_strategy`, with both paths ultimately writing to Pinecone.

---

## ⚙️ Key Components

| Module | Purpose |
|--------|----------|
| **`main.py`** | Streamlit entry point. Initialises cache, handles UI, logging, and session state. |
| **`intent_classifier.py`** | NLPIntentClassifier - CT2 encoder (hf-hub-ctranslate2) with context-aware biasing and calibrated confidence |
| **`query_manager.py`** | Routes user input to the appropriate handler using a priority-based system. Integrates NLPIntentClassifier for hybrid intent pipeline |
| **`query_context.py`** | Encapsulates query metadata (buildings, business terms, complexity) used for context-aware classification |
| **`query_types.py`** | Enum defining all supported query intents (CONVERSATIONAL, MAINTENANCE, RANKING, etc.) |
| **`base_handler.py`** | Abstract base class for all query handlers with consistent logging and metadata extraction. |
| **Handlers Layer** | Specialised query processors implementing `can_handle()` and `handle()` methods: |
| → `conversational_handler.py` | Responds to greetings, about queries, and small talk. |
| → `counting_handler.py` | Handles counting queries ("How many buildings have FRAs?"). |
| → `maintenance_handler.py` | Handles maintenance requests, jobs, and categories. |
| → `property_handler.py` | Handles property condition and derelict building queries. |
| → `ranking_handler.py` | Handles "largest/smallest/top" building queries. |
| → `semantic_search_handler.py` | Fallback search handler for all remaining queries using Pinecone semantic vector retrieval + OpenAI summarisation. |
| **`search_core` package** | Unified structured + semantic retrieval engine |
| → `search_router.py` | Unified entry point for structured and semantic searches. |
| **`search_instructions.py`** | Defines `SearchInstructions` dataclass to pass structured search intent (root-level module, used by search_core). |
| **`search_core` package (cont.)** |  |
| → `planon_search.py` | Handles property and Planon-related structured queries. |
| → `maintenance_search.py` | Handles structured maintenance vector lookups. |
| → `search_utils.py` | Core utilities for boosting, deduplication, and building filters. |
| **`building/utils.py`** | Comprehensive building cache, alias, and fuzzy matching utilities (centralised). |
| **`structured_queries.py`** | Rule-based structured detection for counting, ranking, maintenance, and property queries. |
| **`config/constant.py`** | Constants for environment, models, and routing configuration. |
| **`config/settings.py`** | Environment, API keys, and Pinecone/OpenAI configuration. |

---

## 🧩 Smart Query Routing

Alfred uses a **Chain of Responsibility pattern** via the `QueryManager`:

1. **Preprocessing**: Extracts buildings, business terms, and analyses query complexity
2. **Intent Classification**: NLPIntentClassifier predicts intent with confidence score
3. **Handler Selection**: Each handler declares a `priority` (lower number = higher priority)
4. **Execution**: The `QueryManager` sequentially checks each handler's `can_handle()` method
5. **Fallback**: `SemanticSearchHandler` handles all remaining unclassified queries

Example:
```text
"Hi Alfred" → ConversationalHandler (priority: 1)
"Which buildings have maintenance requests?" → MaintenanceHandler (priority: 2)
"Top 10 largest buildings" → RankingHandler (priority: 3)
"Which buildings are derelict?" → PropertyHandler (priority: 4)
"How many buildings have FRAs?" → CountingHandler (priority: 5)
"Describe frost protection in Berkeley Square" → SemanticSearchHandler (priority: 99)
```

---

## 🧱 search_core Layer

The `search_core` package provides a **unified structured + semantic retrieval system**.

### 🔍 `SearchInstructions`
```python
@dataclass
class SearchInstructions:
    type: str           # "semantic", "planon", "maintenance"
    query: str
    top_k: int
    building: str | None = None
    document_type: str | None = None
```

All handlers construct a `SearchInstructions` object when a search is needed.  
The router then calls the correct backend automatically:

```python
from search_core.search_router import execute
results, answer, pub_date, score_flag = execute(SearchInstructions(
    type="semantic",
    query="Fire Risk Assessment for Senate House",
    top_k=5
))
```

---

## 🗝️ Building Cache & Matching

`building/utils.py` serves as the single source of truth for:

- Alias and canonical name mapping  
- Multi-index cache population  
- Fuzzy matching and validation  
- Building-specific result filtering  
- Metadata filter generation for Pinecone

Building cache initialisation runs at app startup, ensuring that all fuzzy and alias-based matches are available to every handler.

---

## 🚀 Features Summary

- **NLP Intent Classification**: CT2 encoder (hf-hub-ctranslate2) with context-aware biasing
- **Modular Handlers**: Each query type handled by a specialised module  
- **Unified Router**: `search_core` dispatches structured vs. semantic searches
- **Session Manager**: `session_manager` Persists building context for previous user query  
- **Smart Building Cache**: Fuzzy and alias matching across multiple metadata fields  
- **OpenAI + Pinecone Integration**: RAG-style search and summarisation  
- **Logging Pipeline**: Standardised, color-coded INFO logs across all modules  
- **Error Isolation**: Each handler logs and fails gracefully without blocking others  

---

## 🔧 Ingestion Updates (V3)

Recent ingestion changes focus on reliability, idempotency, and observability:

**Core changes**
- **Interfaces layer** for ingestion ports (`VectorStore`, `Embedder`, `EventSink`, `IngestFileRegistry`, `JobRegistry`).
- **Redis-backed registries** for files and jobs, with status/TTL handling and atomic lease semantics.
- **File state machine** with explicit states: discovered → processing → upserted → verified → success/failed.
- **Tokenised processing**: each file run gets a `processing_token` enforced in registry state transitions.
- **VectorStore abstraction** wraps Pinecone calls and normalises error handling.
- **Embedder wrapper** owns retries/backoff/batch splitting and returns explicit index → embedding/error mappings.
- **Unified upsert scheduling** via `VectorWriteCoordinator` (inline or worker strategy).
- **Verification paths** use the VectorStore abstraction; failures emit structured events.

**Metrics & events**
- Prometheus counters for embedding retries, batch reductions, rate limits, upsert timing, lock contention, rollback failures, and FRA supersession update outcomes (bulk vs per-item success/failure).
- JSONL event sink for ingestion summaries and verification alerts.

**Timeouts & safety**
- Configurable OpenAI timeouts (total + connect/read/write/pool) and per-file max wall-clock.
- Queue draining and failed-state recording on worker stop events.

**Redis**
- File records stored as **hashes** (not JSON blobs) with TTLs based on status.
- Job records use SETNX-style semantics to avoid duplicate supersession runs.

---

## 🧰 Developer Guide

### Environment Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```

### Required Environment Variables

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USERNAME=optional
REDIS_PASSWORD=optional
ANSWER_MODEL=gpt-4o-mini
DEFAULT_EMBED_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
OPENAI_TIMEOUT=120
OPENAI_CONNECT_TIMEOUT=10
OPENAI_READ_TIMEOUT=60
OPENAI_WRITE_TIMEOUT=60
OPENAI_POOL_TIMEOUT=30
MAX_FILE_SECONDS=900
```

### Key Dependencies

```
# Core
streamlit==1.49.1
openai>=1.0.0
pinecone>=3.0.0

# NLP + ML
hf-hub-ctranslate2>=2.0.0     # CT2 encoder for intent classification
torch>=2.1.0                  # PyTorch backend for CT2 encoder
textblob==0.19.0             # Spell checking
numpy>=1.24                  # Vector operations
scikit-learn>=1.4.0          # Additional ML utilities
```

### Model Files

The NLPIntentClassifier expects:
- **Local model**: `models/all-MiniLM-L6-v2/` (auto-extracted from .zip if present)
- **Cache**: `intent_embeddings_cache.json` and `intent_embeddings_cache.npz` (auto-generated on first run)
- **Fallback**: Auto-downloads from Hugging Face if local model not found

### Generated files

- intent_embeddings_cache.json and intent_embeddings_cache.npz are generated at runtime and should not be committed

### Logging

- Configured globally in `main.py` using `logging.basicConfig()`
- All handlers inherit logger from `BaseQueryHandler`
- Streamlit environment forced to INFO level with `STREAMLIT_LOG_LEVEL=info`

---

## 🧪 Example Queries

| Query | Predicted Intent | Handler |
|--------|------------------|----------|
| "Hi Alfred" | CONVERSATIONAL | ConversationalHandler |
| "Which buildings have FRAs?" | COUNTING | CountingHandler |
| "Show maintenance for Senate House" | MAINTENANCE | MaintenanceHandler |
| "Top 5 largest buildings by area" | RANKING | RankingHandler |
| "Which buildings are derelict?" | PROPERTY_CONDITION | PropertyHandler |
| "Show the AHU logic in Senate House" | SEMANTIC_SEARCH | SemanticSearchHandler |

---

## 🧩 Design Principles

- **Separation of Concerns** — Handlers only decide *what* to do; search_core decides *how*.  
- **Extensibility** — Add new query handlers (e.g., "EnergyHandler") without touching core logic.  
- **Transparency** — Every query logs its route and detection path.  
- **Consistency** — All results conform to `QueryResult` schema.
- **Context Awareness** — Intent classification considers extracted buildings and business terms.
- **Graceful Degradation** — Falls back to pattern matching if ML model unavailable.

---

## 🔒 Security Features

Alfred implements defence-in-depth security across multiple layers:

### Input Validation (`input_validator.py`)

Comprehensive query validation with:
- **Prompt injection detection**: Blocks attempts to manipulate AI behaviour via malicious inputs
- **DoS protection**: Query length limits and complexity analysis
- **SQL/NoSQL injection patterns**: Detects common injection vectors
- **Unicode normalisation**: Prevents homoglyph attacks
- **Whitelist validation**: Ensures queries contain only permitted characters

```python
from input_validator import validate_query_security, get_validation_summary

result = validate_query_security(user_query)
if not result.is_valid:
    logger.warning(f"Blocked query: {result.rejection_reason}")
```

### File Operations Security (`file_operations_validator.py`)

OWASP-compliant file handling with:
- **Path traversal prevention**: Blocks `../` and absolute path escapes
- **Symlink protection**: Detects and blocks symbolic link attacks
- **File type whitelisting**: Only permits known-safe extensions (`.pdf`, `.docx`, `.xlsx`, etc.)
- **Size limits**: Configurable maximum file sizes
- **Filename sanitisation**: Removes dangerous characters and sequences

```python
from file_operations_validator import (
    validate_path_safety,
    is_safe_extension,
    read_file_safe,
    validate_file_safety
)

# Validate path safety
safe_path = validate_path_safety(base_directory, user_provided_path)

# Check file extension
if is_safe_extension(filename):
    content = read_file_safe(base_directory, relative_path)
```

### Rate Limiting (`rate_limiter.py`)

Redis-backed rate limiting with:
- **Per-user query limits**: Prevents abuse from individual sessions
- **Global rate caps**: Protects against coordinated attacks
- **Sliding window algorithm**: Fair burst handling
- **Configurable thresholds**: Separate limits for queries vs file operations

### Credential Management (`credential_manager.py`)

Secure credential handling:
- **Environment variable isolation**: No hardcoded secrets
- **Lazy loading**: Credentials fetched only when needed
- **Validation on access**: Ensures credentials meet format requirements

### Log Sanitisation (`log_sanitiser.py`)

Prevents sensitive data leakage:
- **PII redaction**: Removes email addresses, phone numbers
- **Credential masking**: Hides API keys and tokens in logs
- **Path normalisation**: Removes user-specific path components

---

## 🔥 FRA (Fire Risk Assessment) Module

The `fra/` package provides structured extraction and prioritisation for Fire Risk Assessments:

### Components

| Component | Purpose |
|-----------|---------|
| `FRAActionPlanParser` | Extracts risk items from FRA PDF documents using regex and structure analysis |
| `FRATriageComputer` | Calculates deterministic priority scores based on risk level, timescale, and category |
| `FRAEnricher` | Enriches extracted items with computed fields (scores, flags, normalised values) |
| `ParsingConfidence` | Tracks extraction reliability per-field and per-document |
| `FRASupersessionHandler` | Manages version control when newer FRAs replace older ones |

### Triage Scoring Algorithm

```python
# Priority score calculation (lower = more urgent)
base_score = RISK_WEIGHTS[risk_level]  # Intolerable=1, Substantial=2, Moderate=3, Tolerable=4
time_modifier = TIMESCALE_WEIGHTS[timescale]  # Immediate=0, 3months=1, 6months=2, 12months=3
category_modifier = CATEGORY_WEIGHTS[category]  # Life safety=0, Compliance=1, Advisory=2

final_score = base_score + (time_modifier * 0.3) + (category_modifier * 0.2)
```

### FRA Vector Structure

Each FRA risk item generates a vector with metadata:
```python
{
    "id": "fra_{building}_{item_number}_{hash}",
    "values": [...],  # 1536-dim embedding
    "metadata": {
        "document_type": "fra_action_item",
        "building": "Senate House",
        "risk_level": "Substantial",
        "timescale": "3 months",
        "category": "Fire doors",
        "priority_score": 2.5,
        "action_required": "Replace fire door seals...",
        "location_detail": "Level 2, Room 2.15",
        "fra_date": "2024-01-15",
        "superseded": false
    }
}
```

---

## 🛠️ Utility Modules

### Session Management (`session_manager.py`)

Persists conversation context across queries:
- **Building context carry-over**: "Tell me more" queries inherit previous building
- **Query history tracking**: Maintains recent query list for context
- **State serialisation**: Streamlit session state management

### Date Utilities (`date_utils.py`)

Intelligent date parsing for document searches:
- **Natural language dates**: "last month", "Q3 2024", "before January"
- **Publication date filtering**: Find documents by date range
- **ISO normalisation**: Consistent date format handling

### Business Terms (`business_terms.py`)

Domain-specific terminology definitions:
```python
BUSINESS_TERMS = {
    "hvac": ["heating", "ventilation", "air conditioning", "ahu", "fcu"],
    "bms": ["building management", "controls", "trend", "bacnet"],
    "fra": ["fire risk", "assessment", "action plan", "risk item"],
    "ppm": ["planned preventive maintenance", "scheduled maintenance"],
    ...
}
```

### Context Sanitisation (`sanitise_context.py`)

Safe rendering for UI output:
- **Markdown escaping**: Prevents injection via search results
- **HTML sanitisation**: Removes potentially dangerous tags
- **Length truncation**: Prevents UI overflow

### Client Management (`clients.py`)

Centralised API client initialisation:
```python
from clients import  ClientManager, get_oai

openai = get_oai()                    # OpenAI client with configured timeouts
redis = ClientManager.get_redis()     # Redis client with connection pooling

```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test module
pytest tests/test_fra_triage.py -v

# Run security tests only
pytest tests/test_file_operations_validator.py tests/test_input_validator.py -v
```

### Test Structure

```
tests/
├── conftest.py                      # Shared fixtures (mock clients, test data)
├── test_file_operations_validator.py # File security validation tests
├── test_fra_triage.py               # FRA triage scoring tests
├── test_intent_classifier.py        # Intent classification tests
├── test_building_utils.py           # Building cache and matching tests
├── test_input_validator.py          # Query validation tests
└── test_search_core.py              # Search router tests
```

### Key Fixtures

```python
@pytest.fixture
def mock_pinecone_index():
    """Returns a mock Pinecone index for testing searches."""
    ...

@pytest.fixture
def sample_fra_document():
    """Returns a sample FRA PDF content for parsing tests."""
    ...

@pytest.fixture
def building_cache():
    """Pre-populated building cache for matching tests."""
    ...
```

---

## 🔐 Security Scanning

### Automated Security Checks

```bash
# Run full security scan
python scripts/security_scan.py --json --strict

# Individual tools
safety check -r requirements.txt
pip-audit
bandit -r . -ll
```

### CI/CD Integration

The `.github/workflows/security-scan.yml` workflow runs on every PR:

1. **Dependency scanning**: `safety` and `pip-audit` check for known vulnerabilities
2. **Static analysis**: `bandit` scans for common security issues
3. **Secret detection**: Checks for accidentally committed credentials
4. **Requirements pinning**: Validates all dependencies are version-pinned

### Security Scan Output

```json
{
    "scan_date": "2025-03-04T10:30:00Z",
    "vulnerabilities_found": 0,
    "warnings": [],
    "checks_passed": ["safety", "pip-audit", "bandit", "pin-check"],
    "recommendations": []
}
```

---

## 🛠️ Tools Directory

Development and debugging utilities in `tools/`:

| Tool | Purpose | Usage |
|------|---------|-------|
| `extract_pdf_text.py` | Extract raw text from PDF files | `python tools/extract_pdf_text.py input.pdf` |
| `parse_goldneyhall_action_plan_from_full_text.py` | Test FRA parsing on sample document | `python tools/parse_goldneyhall_action_plan_from_full_text.py` |
| `profile_intent.py` | Profile intent classifier performance | `python tools/profile_intent.py --queries 1000` |

### PDF Text Extraction

```bash
# Extract text for debugging
python tools/extract_pdf_text.py "path/to/document.pdf" --output extracted.txt

# Extract with page markers
python tools/extract_pdf_text.py "path/to/document.pdf" --page-markers
```

### Intent Profiling

```bash
# Profile classification speed
python tools/profile_intent.py --queries 1000 --warmup 100

# Output:
# Average latency: 2.3ms
# P95 latency: 4.1ms
# Throughput: 435 queries/sec
```

---

## ⚙️ Additional Environment Variables

Beyond the core variables, these optional settings provide fine-grained control:

```bash
# Environment mode
ENVIRONMENT=development          # development | staging | production
IS_PRODUCTION=false             # Enables stricter validation in production

# Feature flags
ENABLE_SERVICE_STATUS=true      # Show service health in UI
ENABLE_RATE_LIMITING=true       # Enable query rate limits
ENABLE_INPUT_VALIDATION=true    # Enable prompt injection detection

# Security settings
MAX_QUERY_LENGTH=2000           # Maximum characters per query
MAX_FILE_SIZE_MB=50             # Maximum upload file size
ALLOWED_FILE_EXTENSIONS=.pdf,.docx,.xlsx,.csv

# Redis settings (optional - falls back to in-memory if unavailable)
REDIS_DB=0                      # Redis database number
REDIS_SSL=false                 # Enable SSL for Redis connection
REDIS_MAX_CONNECTIONS=10        # Connection pool size

# Pinecone settings
PINECONE_ENVIRONMENT=us-east-1  # Pinecone region
PINECONE_INDEX_NAME=alfred-v3   # Index name
PINECONE_NAMESPACE=production   # Namespace for vectors

# Logging
LOG_FORMAT=json                 # json | text
LOG_FILE=/var/log/alfred.log    # Optional file logging
SENSITIVE_LOG_FIELDS=api_key,password,token  # Fields to redact
```

---

## 📊 Metrics & Observability

### Prometheus Metrics

Alfred exposes Prometheus-compatible metrics:

```python
# Ingestion metrics
alfred_embedding_retries_total          # Count of embedding API retries
alfred_embedding_batch_reductions_total # Count of batch size reductions
alfred_rate_limit_hits_total            # OpenAI rate limit encounters
alfred_upsert_duration_seconds          # Histogram of upsert latencies
alfred_lock_contention_total            # Redis lock contention events
alfred_rollback_failures_total          # Failed rollback attempts

# FRA-specific metrics
alfred_fra_supersession_bulk_success    # Successful bulk supersession updates
alfred_fra_supersession_bulk_failure    # Failed bulk supersession updates
alfred_fra_supersession_item_success    # Individual item update successes
alfred_fra_supersession_item_failure    # Individual item update failures

# Query metrics
alfred_query_latency_seconds            # End-to-end query latency
alfred_intent_classification_seconds    # Intent classifier latency
alfred_handler_invocations_total        # Count by handler type
```

### Event Sink (JSONL)

Structured events for audit and debugging:

```json
{"event": "ingestion_complete", "timestamp": "2025-03-04T10:30:00Z", "files_processed": 150, "vectors_upserted": 4523, "errors": 2}
{"event": "verification_alert", "timestamp": "2025-03-04T10:31:00Z", "file": "fra_senate_house.pdf", "expected_vectors": 45, "found_vectors": 43}
{"event": "fra_supersession", "timestamp": "2025-03-04T10:32:00Z", "building": "Senate House", "old_fra_date": "2023-01-15", "new_fra_date": "2024-01-15", "items_superseded": 38}
```

---

## 🗂️ Project Structure

```
Alfred-V3/
├── main.py                     # Streamlit entry point
├── intent_classifier.py        # NLP intent classification
├── query_manager.py            # Query routing orchestrator
├── query_context.py            # Query metadata container
├── query_types.py              # Intent enum definitions
├── search_instructions.py      # Search request dataclass
├── structured_queries.py       # Rule-based query detection
│
├── # Security modules (root level)
├── input_validator.py          # Query validation & injection detection
├── file_operations_validator.py # Path traversal & file safety
├── rate_limiter.py             # Redis-backed rate limiting
├── log_sanitiser.py            # PII redaction in logs
├── credential_manager.py       # Secure credential handling
│
├── # Additional utilities (root level)
├── alfred_exceptions.py        # Custom exception classes
├── pinecone_utils.py           # Pinecone helper functions
├── sanitise_context.py         # Context sanitisation for UI
├── session_manager.py          # Streamlit session management
├── clients.py                  # API client management
├── emojis.py                   # Emoji constants
├── word_to_pdf.py              # Document conversion utility
├── date_utils.py               # Date parsing utilities
├── business_terms.py           # Domain terminology
├── analyse_events_jsonl.py     # Event log analysis
│
├── query_handlers/             # Handler implementations
│   ├── __init__.py
│   ├── base_handler.py
│   ├── conversational_handler.py
│   ├── counting_handler.py
│   ├── maintenance_handler.py
│   ├── property_handler.py
│   ├── ranking_handler.py
│   └── semantic_search_handler.py
│
├── search_core/                # Search engine layer
│   ├── __init__.py
│   ├── search_router.py
│   ├── planon_search.py
│   ├── maintenance_search.py
│   └── search_utils.py
│
├── building/                   # Building data management
│   ├── __init__.py
│   ├── utils.py
│   ├── path_inventory.py
│   ├── path_inventory_summary.py
│   └── alias_override.py
│
├── fra/                        # FRA processing
│   ├── __init__.py
│   ├── parser.py
│   ├── triage.py
│   ├── enricher.py
│   └── supersession.py
│
├── ingest/                     # Ingestion pipeline
│   ├── __init__.py
│   ├── interfaces.py
│   ├── orchestrator.py
│   ├── coordinator.py
│   ├── document_processor.py
│   └── registries.py
│
├── interfaces/                 # Abstract interfaces
│   └── __init__.py             # VectorStore, Embedder, EventSink, etc.
│
├── config/                     # Configuration
│   ├── __init__.py
│   ├── constant.py
│   └── settings.py
│
├── tests/                      # Test suite
│   ├── conftest.py
│   └── test_*.py
│
├── tools/                      # Development utilities
│   ├── extract_pdf_text.py
│   ├── profile_intent.py
│   └── parse_goldneyhall_action_plan_from_full_text.py
│
├── scripts/                    # Operational scripts
│   └── security_scan.py
│
├── models/                     # ML model files
│   └── all-MiniLM-L6-v2/
│
├── .github/
│   └── workflows/
│       └── security-scan.yml
│
├── requirements.txt
├── README.md
└── .gitignore
```
## 📝 License

Internal use only — University of Bristol Smart Buildings Team  
© 2025 University of Bristol