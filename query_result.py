#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QueryResult - the unified output container for all query handlers.

Returned by QueryManager.process_query(), it carries:
    • The final answer string
    • Optional structured results
    • Contextual metadata (buildings, filters, scoring info, etc.)
    • Diagnostics: which handler was used, success state, processing time

Handlers may enrich metadata as needed.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class QueryResult:
    """
    Represents the final response for a processed query.

    Required:
        query (str) - The original user query
        answer (str) - Final answer text produced by a handler

    Optional fields:
        results (list) - Structured search results, documents, rows, etc.
        handler_used (str) - Name of the handler that produced the answer
        query_type (str) - Logical category: 'counting', 'maintenance', etc.
        success (bool) - Indicates whether the handler executed successfully
        publication_date_info (Any) - Optional semantic search metadata
        score_too_low (bool | None) - Embedding score threshold signal
        metadata (dict) - Arbitrary enriched information
    """

    # ------------------------------------------------------------------
    # Core fields
    # ------------------------------------------------------------------
    query: str
    answer: Optional[str]

    # Optional structured results (semantic search, SQL rows, etc.)
    results: List[Any] = field(default_factory=list)

    # Which handler produced the response
    handler_used: Optional[str] = None

    # Query semantic category (string version of QueryType)
    query_type: Optional[str] = None

    # Whether handler execution was successful
    success: bool = True

    # Processing time
    processing_time_ms: Optional[float] = None

    # Semantic search extras
    publication_date_info: Any = None
    score_too_low: Optional[bool] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Guarantee metadata is always a dict
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a single metadata item."""
        self.metadata[key] = value

    def merge_metadata(self, data: Dict[str, Any]) -> None:
        """Merge multiple metadata values safely."""
        if data:
            for k, v in data.items():
                # Handler-level metadata should not overwrite core fields
                if k not in self.metadata:
                    self.metadata[k] = v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to a dict (safe for transport or API output).
        """
        return asdict(self)

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        return (
            f"QueryResult(query={self.query!r}, "
            f"handler={self.handler_used!r}, "
            f"query_type={self.query_type!r}, success={self.success}, "
            f"results={len(self.results)} items)"
        )
