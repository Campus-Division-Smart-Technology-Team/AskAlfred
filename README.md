# 🦍 Alfred V2 — Modular, Hybrid-Intent Building-Aware Search Assistant

Alfred is an intelligent, Streamlit-based search assistant for the University of Bristol’s Campus Innovation Technology team.

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

Alfred V2 uses a **Hybrid Intent Routing System**, implemented in:

### ✅ `intent_classifier.py`
A lightweight, local ML model using **SentenceTransformers** (`all-MiniLM-L6-v2`) that:

- embeds all intent labels at startup (cached for speed)
- vector-matches user queries to intents
- returns both `predicted_intent` and `confidence`
- integrates into `QueryContext` and `QueryManager`

### ✅ New behaviour:
- If `predicted_intent == semantic_search` **and confidence < 0.60**, fallback to RAG  
- If a handler declines during negotiation, QueryManager escalates automatically  
- Old legacy `query_classifier.py` is completely removed

---

## 🧠 Core Architecture Overview

Alfred’s architecture follows a **modular, layered design**:

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
       │ Rule Layer → Regex/Keyword Matching │
       │ ML Layer → NLPIntentClassifier      │
       └───────────────┬────────────────────┘
                       │
    ┌──────────────────▼────────────────────┐
    │         Handlers Layer                │
    │ (Conversational / Property /           │
    │  Maintenance / Counting / Ranking /    │
    │  SemanticSearch)                       │
    └────────────────────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │   search_core package   │
            └────────────────────────┘
```

---

## ⚙️ Key Components

| Module | Purpose |
|--------|----------|
| **`main.py`** | Streamlit entry point. Initialises cache, handles UI, logging, and session state. |
| ** `intent_classifier.py`** | Local ML classifier (SentenceTransformers) with confidence output |
| **`query_manager.py`** | Routes user input to the appropriate handler using a weighted priority system. Hybrid intent pipeline + rule layer + fallback logic |
| **`base_handler.py`** | Abstract base class for all query handlers with consistent logging and metadata extraction. |
| **Handlers Layer** | Specialised query processors implementing `can_handle()` and `handle()` methods: |
| → `conversational_handler.py` | Responds to greetings, about queries, and small talk. |
| → `counting_handler.py` | Handles counting queries (“How many buildings have FRAs?”). |
| → `maintenance_handler.py` | Handles maintenance requests, jobs, and categories. |
| → `property_handler.py` | Handles property condition and derelict building queries. |
| → `ranking_handler.py` | Handles “largest/smallest/top” building queries. |
| → `semantic_search_handler.py` | Fallback search handler for all remaining queries using federated semantic search. |
| **`search_core` package** | The new modular search layer for Unified structured + semantic retrieval engine |
| → `search_router.py` | Unified entry point for structured and semantic searches. |
| → `search_instructions.py` | Defines `SearchInstructions` dataclass to pass structured search intent. |
| → `semantic_search.py` | Runs Pinecone semantic vector retrieval + OpenAI summarization. |
| → `planon_search.py` | Handles property and Planon-related structured queries. |
| → `maintenance_search.py` | Handles structured maintenance vector lookups. |
| → `search_utils.py` | Core utilities for boosting, deduplication, and building filters. |
| → `building_utils.py`** | Comprehensive building cache, alias, and fuzzy matching utilities (centralized). |
| → `structured_queries.py`** | Maintains structured detection for counting, ranking, maintenance, and property queries. |
| → `config.py`** | Global environment, API keys, and Pinecone/OpenAI configuration. |

| → `structured_queries.py`** | Rule-based structured detection (counting, ranking, condition queries) |



---

## 🧩 Smart Query Routing

Alfred uses a **Chain of Responsibility pattern** via the `QueryManager`:

1. Each handler declares a `priority` (lower number = higher priority).
2. The `QueryManager` sequentially checks each handler’s `can_handle()` method.
3. The first handler returning `True` processes the query.
4. Fallback: `SemanticSearchHandler` handles all remaining unclassified queries.

Example:
```text
"Hi Alfred" → ConversationalHandler
"Which buildings have maintenance requests?" → MaintenanceHandler
"Which buildings are derelict?" → PropertyHandler
"Top 10 largest buildings" → RankingHandler
"How many buildings have FRAs?" → CountingHandler
"Describe frost protection in Berkeley Square" → SemanticSearchHandler
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

## 🏗️ Building Cache & Matching

`building_utils.py` now serves as the single source of truth for:

- Alias and canonical name mapping  
- Multi-index cache population  
- Fuzzy matching and validation  
- Building-specific result filtering  
- Metadata filter generation for Pinecone

Building cache initialization runs at app startup, ensuring that all fuzzy and alias-based matches are available to every handler.

---

## 🚀 Features Summary

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

### Logging

- Configured globally in `main.py` using `logging.basicConfig()`
- All handlers inherit logger from `BaseQueryHandler`
- Streamlit environment forced to INFO level with `STREAMLIT_LOG_LEVEL=info`

---

## 🧪 Example Queries

| Query | Handler |
|--------|----------|
| “Hi Alfred” | ConversationalHandler |
| “Which buildings have FRAs?” | CountingHandler |
| “Which buildings have maintenance requests?” | MaintenanceHandler |
| “Which buildings are derelict?” | PropertyHandler |
| “Top 5 largest buildings by area” | RankingHandler |
| “Show the AHU logic in Senate House” | SemanticSearchHandler |

---

## 🧩 Design Principles

- **Separation of Concerns** – Handlers only decide *what* to do; search_core decides *how*.  
- **Extensibility** – Add new query handlers (e.g., “EnergyHandler”) without touching core logic.  
- **Transparency** – Every query logs its route and detection path.  
- **Consistency** – All results conform to `QueryResult` schema.  

---

## 🧱 Migration Notes (from Alfred v1)

| Old Component | Replaced By |
|----------------|-------------|
| `search_operations.py` | ❌ Deprecated → split into `search_core/` modules |
| Inline semantic + planon logic | ✅ Now in `search_router.execute()` |
| `perform_federated_search()` | ✅ Replaced by `SearchInstructions` + unified router |
| Multiple building filters | ✅ Centralized in `building_utils.py` |
| One-file design | ✅ Modular, extensible handler framework |

---

## 📝 License

Internal use only — University of Bristol Smart Technology Team  
© 2025 University of Bristol
