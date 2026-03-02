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

## 📝 License

Internal use only — University of Bristol Smart Technology Team  
© 2025 University of Bristol