# 🦍 Alfred V2 — Modular, Hybrid-Intent Building-Aware Search Assistant

Alfred is an intelligent, Streamlit-based search assistant for the University of Bristol's Campus Innovation Technology team.

It provides **multi-domain, building-aware search** across:
- Building Management Systems (BMS)
- Fire Risk Assessments (FRAs)
- Planon property data (conditions, areas, metadata)
- Maintenance requests and job records  
- General RAG / semantic search across documentation

Powered by:
✅ OpenAI embeddings  
✅ Pinecone vector search  
✅ A hybrid **rule-based + ML intent classifier** pipeline  

---

## 🧠 Intent Detection (New in V2)

Alfred V2 uses a **Hybrid Intent Routing System** with the `NLPIntentClassifier`:

### ✅ `intent_classifier.py` - NLPIntentClassifier
A sophisticated, context-aware intent classifier using **Hugging Face's SentenceTransformers** (`all-MiniLM-L6-v2`) that:

**Core Features:**
- Loads pre-trained model from local `models/all-MiniLM-L6-v2/` directory or auto-downloads from Hugging Face
- Auto-extracts zipped models at startup for convenience
- Generates and caches intent embeddings for all query types (pickled for speed in `intent_embeddings_cache.pkl`)
- Returns calibrated confidence scores using **softmax normalization**
- Provides both semantic and pattern-based classification with automatic fallback

**Advanced Capabilities:**
- **Context-aware biasing**: Adjusts confidence scores based on `QueryContext` (detected buildings, business terms)
- **Hybrid classification**: Combines semantic similarity (70% mean + 30% max example) with pattern matching
- **Confidence threshold**: Default 0.65 threshold triggers pattern fallback for low-confidence predictions
- **Graceful degradation**: Falls back to pattern-only mode if SentenceTransformers unavailable

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

## ⚙️ Key Components

| Module | Purpose |
|--------|----------|
| **`main.py`** | Streamlit entry point. Initialises cache, handles UI, logging, and session state. |
| **`intent_classifier.py`** | NLPIntentClassifier - Hugging Face SentenceTransformers model with context-aware biasing and calibrated confidence |
| **`query_manager.py`** | Routes user input to the appropriate handler using a weighted priority system. Integrates NLPIntentClassifier for hybrid intent pipeline |
| **`query_context.py`** | Encapsulates query metadata (buildings, business terms, complexity) used for context-aware classification |
| **`query_types.py`** | Enum defining all supported query intents (CONVERSATIONAL, MAINTENANCE, RANKING, etc.) |
| **`base_handler.py`** | Abstract base class for all query handlers with consistent logging and metadata extraction. |
| **Handlers Layer** | Specialised query processors implementing `can_handle()` and `handle()` methods: |
| → `conversational_handler.py` | Responds to greetings, about queries, and small talk. |
| → `counting_handler.py` | Handles counting queries ("How many buildings have FRAs?"). |
| → `maintenance_handler.py` | Handles maintenance requests, jobs, and categories. |
| → `property_handler.py` | Handles property condition and derelict building queries. |
| → `ranking_handler.py` | Handles "largest/smallest/top" building queries. |
| → `semantic_search_handler.py` | Fallback search handler for all remaining queries using federated semantic search. |
| **`search_core` package** | Unified structured + semantic retrieval engine |
| → `search_router.py` | Unified entry point for structured and semantic searches. |
| → `search_instructions.py` | Defines `SearchInstructions` dataclass to pass structured search intent. |
| → `semantic_search.py` | Runs Pinecone semantic vector retrieval + OpenAI summarization. |
| → `planon_search.py` | Handles property and Planon-related structured queries. |
| → `maintenance_search.py` | Handles structured maintenance vector lookups. |
| → `search_utils.py` | Core utilities for boosting, deduplication, and building filters. |
| **`building_utils.py`** | Comprehensive building cache, alias, and fuzzy matching utilities (centralized). |
| **`structured_queries.py`** | Rule-based structured detection for counting, ranking, maintenance, and property queries. |
| **`config.py`** | Global environment, API keys, and Pinecone/OpenAI configuration. |

---

## 🧩 Smart Query Routing

Alfred uses a **Chain of Responsibility pattern** via the `QueryManager`:

1. **Preprocessing**: Extracts buildings, business terms, and analyzes query complexity
2. **Intent Classification**: NLPIntentClassifier predicts intent with confidence score
3. **Handler Selection**: Each handler declares a `priority` (lower number = higher priority)
4. **Execution**: The `QueryManager` sequentially checks each handler's `can_handle()` method
5. **Fallback**: `SemanticSearchHandler` handles all remaining unclassified queries

Example:
```text
"Hi Alfred" → ConversationalHandler (priority: 1)
"Which buildings have maintenance requests?" → MaintenanceHandler (priority: 2)
"Which buildings are derelict?" → PropertyHandler (priority: 3)
"Top 10 largest buildings" → RankingHandler (priority: 4)
"How many buildings have FRAs?" → CountingHandler (priority: 5)
"Describe frost protection in Berkeley Square" → SemanticSearchHandler (priority: 99)
```

---

## 🧱 search_core Layer

The new `search_core` package provides a **unified structured + semantic retrieval system**.

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

`building_utils.py` now serves as the single source of truth for:

- Alias and canonical name mapping  
- Multi-index cache population  
- Fuzzy matching and validation  
- Building-specific result filtering  
- Metadata filter generation for Pinecone

Building cache initialization runs at app startup, ensuring that all fuzzy and alias-based matches are available to every handler.

---

## 🚀 Features Summary

- **NLP Intent Classification**: Hugging Face SentenceTransformers with context-aware biasing
- **Modular Handlers**: Each query type handled by a specialized module  
- **Unified Router**: `search_core` dispatches structured vs. semantic searches  
- **Smart Building Cache**: Fuzzy and alias matching across multiple metadata fields  
- **OpenAI + Pinecone Integration**: RAG-style search and summarization  
- **Logging Pipeline**: Standardized, color-coded INFO logs across all modules  
- **Error Isolation**: Each handler logs and fails gracefully without blocking others  

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
ANSWER_MODEL=gpt-4o-mini
DEFAULT_EMBED_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
```

### Key Dependencies

```
# Core
streamlit==1.49.1
openai>=1.0.0
pinecone>=3.0.0

# NLP + ML
sentence-transformers==2.7.0  # Hugging Face transformers for intent classification
torch>=2.1.0                  # PyTorch backend for SentenceTransformers
textblob==0.19.0             # Spell checking
numpy>=1.24                  # Vector operations
scikit-learn>=1.4.0          # Additional ML utilities
```

### Model Files

The NLPIntentClassifier expects:
- **Local model**: `models/all-MiniLM-L6-v2/` (auto-extracted from .zip if present)
- **Cache**: `intent_embeddings_cache.pkl` (auto-generated on first run)
- **Fallback**: Auto-downloads from Hugging Face if local model not found

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
| "Which buildings are derelict?" | PROPERTY_CONDITION | PropertyHandler |
| "Top 5 largest buildings by area" | RANKING | RankingHandler |
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

## 🧱 Migration Notes (from Alfred v1)

| Old Component | Replaced By |
|----------------|-------------|
| `search_operations.py` | ❌ Deprecated → split into `search_core/` modules |
| `query_classifier.py` | ❌ Removed → replaced by `NLPIntentClassifier` |
| Inline semantic + planon logic | ✅ Now in `search_router.execute()` |
| `perform_federated_search()` | ✅ Replaced by `SearchInstructions` + unified router |
| Multiple building filters | ✅ Centralized in `building_utils.py` |
| One-file design | ✅ Modular, extensible handler framework |
| Simple keyword matching | ✅ Hugging Face SentenceTransformers with context biasing |

---

## 📝 License

Internal use only — University of Bristol Smart Technology Team  
© 2025 University of Bristol