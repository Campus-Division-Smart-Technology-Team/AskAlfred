#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved MaintenanceHandler.
Handles maintenance-related requests, jobs, categories, and metrics.
Delegates logic to counting_queries.generate_maintenance_answer.
"""
from query_types import QueryType
from query_context import QueryContext
from query_result import QueryResult

from structured_queries import (
    is_maintenance_query,
    is_ranking_query,
    is_property_condition_query,
    is_counting_query,
)

from .base_handler import BaseQueryHandler


class MaintenanceHandler(BaseQueryHandler):
    """Handles maintenance requests, categories, and maintenance job lookup queries."""

    def __init__(self):
        super().__init__()
        self.query_type = QueryType.MAINTENANCE
        self.priority = 2

    def can_handle(self, context: QueryContext) -> bool:
        """
        MaintenanceHandler should only activate for genuine maintenance questions.
        Avoid overlaps with:
          - ranking queries (e.g., "largest backlog")
          - property condition queries
          - general counting queries
        """

        q = context.query.strip().lower()

        if is_maintenance_query(q):
            return True

        # Avoid conflicts with other handlers
        if is_ranking_query(q):
            return False
        if is_property_condition_query(q):
            return False
        # if is_counting_query(q):
        #     return False

        # Use the precise logic from counting_queries
        return False

    def _as_name(self, building):
        if not building:
            return None
        return getattr(building, "name", None) or str(building)

    def handle(self, context: QueryContext) -> QueryResult:
        """Produce structured maintenance information via structured_queries logic."""
        self._log_handling(context)
        query_text = context.query.strip()

        try:
            from structured_queries import generate_maintenance_answer

            prev_building = None
            previous_context = getattr(context, "previous_context", None)
            if previous_context:
                prev_building = previous_context.get("building")

            building_override = (
                self._as_name(context.building)
                or context.building_filter
                or prev_building
            )

            answer = generate_maintenance_answer(
                query_text, building_override=building_override)

            if not answer:
                answer = (
                    "I couldn't identify any maintenance information for your query. "
                    "You can try specifying a building, maintenance category, or job type."
                )

            return QueryResult(
                query=query_text,
                answer=answer,
                results=[],
                handler_used="MaintenanceHandler",
                query_type=self.query_type.value,
                metadata={"structured_response": True},
            )

        except Exception as e:
            self.logger.error(f"Maintenance handler error: {e}", exc_info=True)

            return QueryResult(
                query=query_text,
                answer="Sorry — I encountered an error while processing your maintenance query.",
                results=[],
                success=False,
                handler_used="MaintenanceHandler",
                query_type=self.query_type.value,
                metadata={"error": str(e)},
            )
