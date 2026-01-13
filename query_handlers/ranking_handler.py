#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RankingHandler.
Handles area-based building ranking queries (largest, smallest, top N, sort by area).
Delegates ranking logic to counting_queries.generate_ranking_answer.
"""

import re
from query_types import QueryType
from query_context import QueryContext
from query_result import QueryResult


from structured_queries import (
    is_ranking_query,
    is_maintenance_query,
    is_property_condition_query,
    is_counting_query,
)

from .base_handler import BaseQueryHandler


class RankingHandler(BaseQueryHandler):
    """Handles building ranking queries (e.g., largest buildings by area)."""

    def __init__(self):
        super().__init__()
        self.query_type = QueryType.RANKING
        self.priority = 3

        # These patterns detect ranking intent specifically about building size/area.
        # More permissive ranking detection is in counting_queries.is_ranking_query,
        # but this handler requires explicit area-related phrasing.
        self.patterns = [
            re.compile(r"\brank(?:ing)?\s+buildings?\b", re.IGNORECASE),
            re.compile(r"\btop\s+\d+\s+buildings?\b", re.IGNORECASE),
            re.compile(
                r"\b(top|largest|biggest|smallest)\s+buildings?\b", re.IGNORECASE),
            re.compile(r"\bsort\s+buildings?\s+by\b", re.IGNORECASE),
        ]

        # Area context words — ensures the query is actually about area ranking,
        # not “largest maintenance backlog” or “biggest problem”.
        self.area_indicators = [
            "area", "size", "gross", "net", "sqm", "square metre", "square meter"
        ]

    def _mentions_area(self, text: str) -> bool:
        """Check if the query explicitly refers to area/size metrics."""
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.area_indicators)

    def can_handle(self, context: QueryContext) -> bool:
        """
        RankingHandler only processes:
          • building area rankings
          • NOT maintenance queries
          • NOT property condition queries
          • NOT document/metadata counting queries
        """

        q = context.query.strip().lower()

        # 1. Avoid overlaps with other handlers
        if is_maintenance_query(q):
            return False
        if is_property_condition_query(q):
            return False
        if is_counting_query(q):
            return False

        # Accept ranking-intent phrase
        if any(p.search(q) for p in self.patterns):
            return True

        if any(word in q for word in ["largest", "biggest", "smallest", "tallest", "highest", "widest", "longest", "most spacious"]):
            return True

        # 2. Basic ranking signals
        if any(p.search(q) for p in self.patterns):
            # Only handle if it’s area-based
            return self._mentions_area(q)

        # 3. Fallback to counting_queries ranking detection,
        # but again, require explicit area context.
        if is_ranking_query(q) and self._mentions_area(q):
            return True

        return False

    def handle(self, context: QueryContext) -> QueryResult:
        """Produce a structured ranking answer via counting_queries."""
        self._log_handling(context)
        query_text = context.query.strip()

        try:
            from structured_queries import generate_ranking_answer

            answer = generate_ranking_answer(query_text)

            if not answer:
                answer = (
                    "I couldn't generate a ranking for your query. "
                    "Try specifying whether you want gross or net area."
                )

            return QueryResult(
                query=query_text,
                answer=answer,
                results=[],
                handler_used="RankingHandler",
                query_type=self.query_type.value,
                metadata={"structured_response": True},
            )

        except Exception as e:
            self.logger.error("Ranking handler error: %s", e, exc_info=True)

            return QueryResult(
                query=query_text,
                answer="Sorry — I encountered an error while generating the ranking.",
                results=[],
                success=False,
                handler_used="RankingHandler",
                query_type=self.query_type.value,
                metadata={"error": str(e)},
            )
