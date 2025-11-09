# search_core/__init__.py

"""
Unified search core package.

Provides:
- semantic_search()      vector semantic retrieval
- planon_search()        structured property/condition/ranking logic
- maintenance_search()   structured maintenance lookups
- execute()              unified router for SearchInstructions
"""

from .semantic_search import semantic_search
from .planon_search import planon_search
from .maintenance_search import maintenance_search

# Router for SearchInstructions
from .search_router import execute

# Utilities (optional re-export)
from .search_utils import (
    search_one_index,
    deduplicate_results,
    apply_doc_type_boost,
    apply_building_boost,
    get_effective_score,
)

__all__ = [
    "semantic_search",
    "planon_search",
    "maintenance_search",
    "execute",
    "search_one_index",
    "deduplicate_results",
    "apply_doc_type_boost",
    "apply_building_boost",
    "get_effective_score",
]
