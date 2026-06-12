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

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from query_core.query_types import QueryType

DENY_ALL_TENANT_ID = "__deny_access__"


def build_access_filter(
    *,
    tenant_id: Optional[str],
    user_roles: tuple[str, ...],
    authenticated: bool,
) -> dict[str, Any]:
    """
    Build the first-pass retrieval access filter from the current auth context.

    This initial enforcement is intentionally narrow: authenticated users are
    constrained to their tenant (and to documents allowing one of their roles,
    when roles are present), while anonymous/dev sessions keep the current
    unfiltered behaviour until the wider authz rollout is complete.

    WARNING (rollout semantics): an authenticated user WITHOUT roles is only
    tenant-scoped and therefore sees role-restricted documents that a user
    WITH non-matching roles cannot — i.e. holding a role can only narrow
    access. This is a deliberate transitional posture (locked in by
    tests/test_access_filter_threading.py) so tenants whose tokens carry no
    app-role claims are not locked out mid-rollout. Once role assignment is
    universal, roleless authenticated users should fail closed instead.
    """
    if not authenticated:
        return {}

    if not tenant_id:
        return {"tenant_id": {"$eq": DENY_ALL_TENANT_ID}}

    access_filter: dict[str, Any] = {"tenant_id": {"$eq": str(tenant_id)}}
    roles = [str(role) for role in user_roles if role]
    if roles:
        access_filter = {
            "$and": [
                access_filter,
                {"allowed_roles": {"$in": roles}},
            ]
        }
    return access_filter


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
    history: Optional[list[dict[str, Any]]] = None
    rolling_summary: Optional[str] = None
    user_id: str = "anonymous"
    user_name: Optional[str] = None
    tenant_id: Optional[str] = None
    user_roles: tuple[str, ...] = field(default_factory=tuple)
    authenticated: bool = False
    auth_source: str = "anonymous"
    # None means "not yet built" (QueryManager builds it from the auth context);
    # an explicit {} means "deliberately unfiltered" and is preserved.
    access_filter: Optional[dict[str, Any]] = None

    # Preprocessor-enriched attributes
    building: Optional[str] = None
    buildings: list[str] = field(default_factory=list)
    business_terms: list[dict[str, Any]] = field(default_factory=list)
    document_type: Optional[str] = None
    complexity: Optional[str] = None
    corrected_query: Optional[str] = None

    # Internal scratchpad
    cache: dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)

    # ML intent (router enrichment)
    predicted_intent: Optional[QueryType] = None
    ml_intent_confidence: float = 0.0
    routing_notes: list[str] = field(default_factory=list)
    # Previous query memory (restored from SessionManager)
    previous_context: Optional[dict[str, Any]] = None
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
            f"query={self.query!r}, user_id={self.user_id!r}, building={self.building!r}, "
            f"document_type={self.document_type!r}, complexity={self.complexity!r}, "
            f"prev_intent={self.previous_intent!r})"
        )
