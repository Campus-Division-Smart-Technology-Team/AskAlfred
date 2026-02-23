#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default semantic search handler.
Handles all remaining queries not claimed by other handlers.
"""

import time
import logging
from typing import Optional, cast, Tuple, Any, List, Dict
from query_types import QueryType
from query_context import QueryContext
from query_result import QueryResult
from search_instructions import SearchInstructions

import search_core
from search_core.search_router import execute

from .base_handler import BaseQueryHandler


class SemanticSearchHandler(BaseQueryHandler):
    """Fallback handler performing federated semantic search."""

    def __init__(self):
        super().__init__()
        self.query_type = QueryType.SEMANTIC_SEARCH
        self.priority = 99
        self.min_query_length = 2          # min words required
        self.min_char_length = 4           # min characters to avoid noise
        self.timeout_seconds = 12          # guardrail for heavy searches

    def can_handle(self, context: QueryContext) -> bool:
        """
        This is the fallback handler,
        but we apply a soft check to avoid meaningless semantic lookups.
        """
        q = context.query.strip()

        # Prevent semantic search on extremely short fragments
        if len(q) < self.min_char_length:
            return True

        if len(q.split()) < self.min_query_length:
            return True

        return True

    def handle(self, context: QueryContext) -> QueryResult:
        """Run federated semantic search with safety guards."""
        instructions: Optional[SearchInstructions] = context.get_from_cache(
            "search_instructions")

        # If a handler provided explicit search instructions, run them
        if instructions:
            return self._execute_instructions(context, instructions)

        ml_intent = getattr(context, "predicted_intent", None)
        ml_conf = getattr(context, "ml_intent_confidence", 0.0)
        if ml_intent:
            # light-touch hint; you could thread this into your search router or boosts
            context.add_to_cache("ml_intent_hint", {"intent": ml_intent.value if hasattr(ml_intent, "value") else str(ml_intent),
                                                    "confidence": ml_conf})

        self._log_handling(context)
        query_text = context.query.strip()

        # Soft handling of very short queries
        if len(query_text) < self.min_char_length:
            return QueryResult(
                query=query_text,
                answer="Could you tell me a bit more? I need a little more detail to search properly.",
                results=[],
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={"short_query": True}
            )

        if len(query_text.split()) < self.min_query_length:
            return QueryResult(
                query=query_text,
                answer="Just a few more words would help me understand what you're looking for.",
                results=[],
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={"short_query": True}
            )

        # Run federated search with timeout
        start = time.time()

        try:
            results, answer, pub_date_info, score_too_low = search_core.semantic_search(
                query_text,
                context.top_k,
                building_filter=context.building_filter
            )

            elapsed = round(time.time() - start, 3)

            # ------------------------------------------------------------
            #  PATCH: Guarantee non-empty answer
            # ------------------------------------------------------------
            if not answer or not isinstance(answer, str) or not answer.strip():
                if results:
                    answer = (
                        f"I found {len(results)} relevant results for your query. "
                        "Let me know if you'd like a summary or details."
                    )
                else:
                    answer = (
                        "I couldn't find any matching information for your query. "
                        "Try rephrasing or adding more detail."
                    )
            # ------------------------------------------------------------
            return QueryResult(
                query=query_text,
                answer=answer,
                results=results,
                publication_date_info=pub_date_info,
                score_too_low=score_too_low,
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={
                    "num_results": len(results),
                    "elapsed_seconds": elapsed,
                    "score_too_low": score_too_low,
                    "building_filter": context.building_filter
                }
            )

        except Exception as e:
            logging.error("Semantic search failure: %s", e, exc_info=True)
            elapsed = round(time.time() - start, 3)

            return QueryResult(
                query=query_text,
                answer="Sorry — something went wrong during semantic search.",
                results=[],
                success=False,
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "fallback": True
                }
            )

    def _execute_instructions(self, context: QueryContext, instr: SearchInstructions) -> QueryResult:
        """Execute a structured search instruction coming from another handler and handles semantic instructions from QueryManager."""
        start = time.time()

        try:
            # Use the unified router
            result = execute(instr)

            if instr.type == "semantic":
                results, answer, pub_info, score_flag = cast(
                    Tuple[List[Dict[str, Any]], str, str, bool],
                    result
                )

            elif instr.type == "planon":
                results, answer, pub_info = cast(
                    Tuple[List[Dict[str, Any]], Optional[str], str],
                    result
                )
                score_flag = None

            elif instr.type == "maintenance":
                results, answer = cast(
                    Tuple[List[Dict[str, Any]], Optional[str]],
                    result
                )
                pub_info = None
                score_flag = None

            else:
                raise ValueError(
                    f"Unknown search instruction type: {instr.type}")

            elapsed = round(time.time() - start, 3)

            return QueryResult(
                query=context.query,
                answer=answer,
                results=results,
                publication_date_info=pub_info,
                score_too_low=score_flag,
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={
                    "instruction_type": instr.type,
                    "elapsed_seconds": elapsed,
                    "building_filter": instr.building,
                    "num_results": len(results),
                }
            )

        except Exception as e:
            logging.error("Search instruction failed: %s", e, exc_info=True)
            elapsed = round(time.time() - start, 3)

            return QueryResult(
                query=context.query,
                answer="Sorry, something went wrong while retrieving the data.",
                results=[],
                success=False,
                handler_used="SemanticSearchHandler",
                query_type=self.query_type.value,
                metadata={
                    "instruction_type": instr.type if instr else None,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                    "fallback": True
                }
            )
