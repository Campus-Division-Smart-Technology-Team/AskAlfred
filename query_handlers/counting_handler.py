#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Delegates counting logic to counting_queries.generate_counting_answer
and avoids overlap with maintenance, ranking, or property-condition routing.
"""

import re
from query_types import QueryType
from query_context import QueryContext
from query_result import QueryResult

from .base_handler import BaseQueryHandler

from structured_queries import (
    is_counting_query,
    is_maintenance_query,
    is_ranking_query,
    is_property_condition_query,
)


class CountingHandler(BaseQueryHandler):
    """Handles pure counting queries (e.g., 'how many buildings…')."""

    def __init__(self):
        super().__init__()
        self.query_type = QueryType.COUNTING
        self.priority = 5

        # Restrictive patterns that *only* identify pure counting intent
        self.patterns = [
            re.compile(r"\bhow\s+many\s+buildings?\b", re.IGNORECASE),
            re.compile(r"\bcount\s+(?:the\s+)?buildings?\b", re.IGNORECASE),
            re.compile(r"\bnumber\s+of\s+buildings?\b", re.IGNORECASE),
            re.compile(r"\blist\s+all\s+buildings?\b", re.IGNORECASE),
        ]

    def can_handle(self, context: QueryContext) -> bool:
        """
        Only handle counting queries that are NOT:
        - maintenance queries
        - ranking queries
        - property condition queries

        Those are routed by their respective handlers.
        """
        q = context.query.strip()

        # Explicitly avoid conflicts with other handlers
        if is_maintenance_query(q):
            return False
        if is_ranking_query(q):
            return False
        if is_property_condition_query(q):
            return False

        # Use both pattern matching and structured_queries detection
        if any(p.search(q.lower()) for p in self.patterns):
            return True

        # Secondary check: allow counting_queries to confirm intent
        return is_counting_query(q)

    def handle(self, context: QueryContext) -> QueryResult:
        """Produce a structured counting answer using counting_queries."""
        self._log_handling(context)
        query_text = context.query.strip()

        try:
            from structured_queries import generate_counting_answer

            answer = generate_counting_answer(query_text)

            if not answer:
                answer = (
                    "I couldn't determine what to count in your query. "
                    "Try asking about buildings, document types, or maintenance data."
                )

            return QueryResult(
                query=query_text,
                answer=answer,
                results=[],
                handler_used="CountingHandler",
                query_type=self.query_type.value,
                metadata={"structured_response": True},
            )

        except Exception as e:
            self.logger.error(f"Counting handler error: {e}", exc_info=True)

            return QueryResult(
                query=query_text,
                answer="Sorry — I ran into a problem while processing your counting request.",
                results=[],
                success=False,
                handler_used="CountingHandler",
                query_type=self.query_type.value,
                metadata={"error": str(e)},
            )
