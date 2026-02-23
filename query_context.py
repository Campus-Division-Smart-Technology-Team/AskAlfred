#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QueryContext - Holds all query-related information during processing.

It is passed through:
    • Preprocessors (BuildingExtractor, BusinessTermExtractor, etc.)
    • Handler routing (QueryManager)
    • Handlers (Counting, Ranking, Maintenance, etc.)

Preprocessors enrich this object; handlers consume it.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import time
from query_types import QueryType


@dataclass
class QueryContext:
    """
    Represents all relevant state for a single user query.

    Attributes filled by QueryManager:
        query (str): Raw user query.
        created_at (float): Timestamp when context was created.
        top_k (int): Number of results requested by the user (e.g. for semantic search).
        building_filter (str | None): Explicit building filter passed by the user.
        cache (dict): Internal scratchpad for preprocessors & handlers.

    Attributes enriched by preprocessors:
        building (str | None)
        business_terms (list)
        document_type (str | None)
        complexity (str)
        corrected_query (str | None)

    Attributes enriched by handlers:
        (optional) anything added via context.add_to_cache()
    """

    # Required user fields
    query: str
    top_k: int = 10
    building_filter: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    rolling_summary: Optional[str] = None

    # Preprocessor-enriched attributes
    building: Optional[str] = None
    buildings: List[str] = field(default_factory=list)
    business_terms: List[Dict[str, Any]] = field(default_factory=list)
    document_type: Optional[str] = None
    complexity: Optional[str] = None
    corrected_query: Optional[str] = None

    # Internal scratchpad
    cache: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)

    # ML intent (router enrichment)
    predicted_intent: Optional[QueryType] = None
    ml_intent_confidence: float = 0.0
    routing_notes: List[str] = field(default_factory=list)
    # Previous query memory (restored from SessionManager)
    previous_context: Optional[Dict[str, Any]] = None
    previous_intent: Optional[str] = None
    previous_intent_confidence: Optional[float] = None

    # ----------------------------------------------------------------------
    # Context helper methods
    # ----------------------------------------------------------------------

    def add_to_cache(self, key: str, value: Any) -> None:
        """Store arbitrary metadata used by preprocessors or handlers."""
        self.cache[key] = value

    def get_from_cache(self, key: str, default: Any = None) -> Any:
        """Retrieve cached information."""
        return self.cache.get(key, default)

    def update_query(self, new_query: str) -> None:
        """
        Used primarily by SpellChecker or normalisation preprocessors.
        Records previous queries for debugging.
        """
        self.add_to_cache("previous_query", self.query)
        self.query = new_query
        self.corrected_query = new_query

    def has_business_term(self, term_type: Optional[str] = None) -> bool:
        """
        Returns True if any business term was extracted,
        optionally filtered by term type (e.g., document_type="FRA").
        """
        if not self.business_terms:
            return False

        if term_type is None:
            return True

        return any(t.get("type") == term_type for t in self.business_terms)

    def __repr__(self) -> str:
        return (
            f"QueryContext("
            f"query={self.query!r}, building={self.building!r}, "
            f"document_type={self.document_type!r}, complexity={self.complexity!r}, "
            f"prev_intent={self.previous_intent!r})"
        )
