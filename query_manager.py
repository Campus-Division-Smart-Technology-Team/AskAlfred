#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Manager - Centralised query orchestration for AskAlfred.

"""


from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import logging
import time


from query_context import QueryContext
from query_route import QueryRoute
from query_result import QueryResult

from query_handlers import (
    ConversationalHandler,
    MaintenanceHandler,
    RankingHandler,
    PropertyHandler,
    CountingHandler,
    SemanticSearchHandler,
)
from query_preprocessors import (
    BuildingExtractor,
    BusinessTermExtractor,
    QueryComplexityAnalyser,
    SpellCheckPreprocessor)

from intent_classifier import NLPIntentClassifier
from query_types import QueryType


# ============================================================================
# QUERY MANAGER
# ============================================================================

class QueryManager:
    """
    Orchestrates the full query lifecycle:
      • Build QueryContext
      • Run preprocessors
      • Execute handler chain
      • Cache responses
      • Track performance stats
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config (dict | None):
                Optional handler configuration. If None, default handlers are used.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Build handler list
        if config:
            self.handlers = self._load_handlers_from_config(config)
        else:
            self.handlers = self._initialise_default_handlers()

        # Sort handlers by priority (lower = higher priority)
        self.handlers.sort(key=lambda h: h.priority)

        # Preprocessors
        self.preprocessors = self._initialise_preprocessors()

        # Cache
        self.cache_enabled = True
        self.cache: Dict[str, QueryResult] = {}

        # Stats
        self.stats = {
            "handlers": {},       # per-handler stats
            "query_types": {},    # per QueryType stats
            "total_queries": 0,
            "overall_total_ms": 0.0,
            "cached_queries": 0
        }

        self.intent_clf = NLPIntentClassifier()

        # Map QueryType
        self.intent_to_handler = {}
        for h in self.handlers:
            if getattr(h, "query_type", None) is not None:
                self.intent_to_handler[h.query_type] = h

    # =========================================================================
    # Handler initialisation
    # =========================================================================

    def _initialise_default_handlers(self) -> List:
        """Create the default handler chain."""
        return [
            ConversationalHandler(),
            MaintenanceHandler(),
            RankingHandler(),
            PropertyHandler(),
            CountingHandler(),
            SemanticSearchHandler(),
        ]

    def _load_handlers_from_config(self, config: Dict) -> List:
        """
        Load handlers from config. Config format:

            {
              "ConversationalHandler": {"enabled": True},
              "RankingHandler": {"enabled": True}
            }

        Missing/disabled handlers are skipped.
        """
        handlers = []

        for handler_name, settings in config.items():
            if not settings.get("enabled", True):
                continue

            try:
                handler_cls = globals().get(handler_name)
                if handler_cls:
                    handlers.append(handler_cls())
                else:
                    self.logger.warning(f"Unknown handler: {handler_name}")
            except Exception as e:
                self.logger.error(f"Could not load {handler_name}: {e}")

        return handlers
    # =========================================================================
    # Preprocessor initialisation
    # =========================================================================

    def _initialise_preprocessors(self) -> List:
        """
        Preprocessors run before handlers and enrich QueryContext.
        Order matters.
        """
        return [
            SpellCheckPreprocessor(),         # Optional: disabled by default
            BuildingExtractor(),
            BusinessTermExtractor(),
            QueryComplexityAnalyser(),
        ]

    # =========================================================================
    # Main processing pipeline
    # =========================================================================

    def process_query(self, query: str, **kwargs) -> QueryResult:
        """
        Main entry point.

        Args:
            query (str): The user query.
            kwargs: Passed to QueryContext (e.g., top_k, building_filter).

        Returns:
            QueryResult
        """
        self.logger.warning(f"RAW QUERY: {query}")

        start_time = time.time()   # timing for this request

        # Create context
        context = QueryContext(query=query, **kwargs)

        # Cache check
        cache_key = self._make_cache_key(context)
        if self.cache_enabled and cache_key in self.cache:
            self.stats["cached_queries"] += 1

            result = self.cache[cache_key]
            elapsed_ms = (time.time() - start_time) * 1000

            # Record telemetry for cached responses (coerce None -> "unknown")
            self._update_stats(
                handler_name=result.handler_used or "unknown",
                query_type=result.query_type or "unknown",
                elapsed_ms=elapsed_ms,
                success=result.success
            )

            result.processing_time_ms = elapsed_ms
            return result

        # ---------------------------------------------------
        # Preprocessors
        # ---------------------------------------------------
        self._run_preprocessors(context)

        self.logger.warning(f"FINAL QUERY BEFORE ROUTING: {context.query!r}")

        # ---------------------------------------------------
        # Routing
        # ---------------------------------------------------
        route = self._route_query_hybrid(context)

        # ---------------------------------------------------
        # Execute handler
        # ---------------------------------------------------
        handler_start = time.time()
        result = route.handler.handle(context)
        handler_elapsed_ms = (time.time() - handler_start) * 1000

        # Attach handler metadata
        result.handler_used = route.handler.__class__.__name__
        result.query_type = route.handler.query_type.value

        if isinstance(route.metadata, dict):
            result.metadata.update(route.metadata)

        # ---------------------------------------------------
        # Stats update (expanded telemetry)
        # ---------------------------------------------------
        self._update_stats(
            handler_name=result.handler_used or "unknown",
            query_type=result.query_type or "unknown",
            elapsed_ms=handler_elapsed_ms,
            success=result.success
        )

        # Cache result
        if self.cache_enabled:
            self.cache[cache_key] = result

        # Total round-trip time for everything
        total_elapsed_ms = (time.time() - start_time) * 1000
        result.processing_time_ms = total_elapsed_ms

        return result

    # =========================================================================
    # Preprocessor execution
    # =========================================================================

    def _run_preprocessors(self, context: QueryContext):
        """Run all preprocessors in order."""
        for pre in self.preprocessors:
            try:
                if pre.should_run(context):
                    pre.process(context)
            except Exception as e:
                self.logger.error(
                    "Preprocessor %s failed: %s", pre.__class__.__name__, e,
                    exc_info=True)

    # =========================================================================
    # Query routing
    # =========================================================================

    def _route_query(self, context: QueryContext) -> QueryRoute:
        """
        Select the best handler using a priority-first strategy.
        """
        best_handler = None
        best_priority = float('inf')

        for handler in self.handlers:
            try:
                if handler.can_handle(context):
                    if handler.priority < best_priority:
                        best_priority = handler.priority
                        best_handler = handler
            except Exception as e:
                self.logger.error(
                    f"Handler {handler.__class__.__name__} failed during can_handle(): {e}",
                    exc_info=True
                )

        # Fallback
        if best_handler is None:
            for h in self.handlers:
                if isinstance(h, SemanticSearchHandler):
                    best_handler = h
                    break

        return QueryRoute(handler=best_handler, metadata={})

    def _route_query_hybrid(self, context: QueryContext) -> QueryRoute:
        """
        Option D (Hybrid) routing:

        1) Rule layer: try handlers' can_handle() (clear-cut cases)
        2) ML classifier: predict intent + confidence for ambiguous cases
        3) Thresholds: if conf < 0.6 -> SemanticSearch with intent in context
        4) Negotiation: chosen handler can still reject; then fallback to SemanticSearch
        """
        # -----------------------------
        # 1) RULE LAYER (handlers)
        # -----------------------------
        best_handler = None
        best_priority = float('inf')

        for handler in self.handlers:
            try:
                if handler.can_handle(context):
                    if handler.priority < best_priority:
                        best_priority = handler.priority
                        best_handler = handler
            except Exception as e:
                self.logger.error(
                    f"Handler {handler.__class__.__name__} failed during can_handle(): {e}",
                    exc_info=True
                )

        # --------------------------------------------------------
        # RULE OVERRIDE LOGIC
        # "Rule layer wins UNLESS ML predicts semantic_search
        #   with high confidence (≥ 0.75)"
        # --------------------------------------------------------
        RULE_OVERRIDE_THRESHOLD = 0.75

        if best_handler and best_handler.__class__.__name__ != "SemanticSearchHandler":
            # Defer ML override check until ML is computed
            rule_candidate = best_handler
        else:
            rule_candidate = None

        # ----------------------------------
        # 2) ML CLASSIFIER for ambiguous case
        # ----------------------------------
        try:
            ml = self.intent_clf.classify_intent(context.query, context)
            context.predicted_intent = ml.intent
            context.ml_intent_confidence = ml.confidence
            self.logger.info(
                "ML intent: %s (%.2f)", ml.intent.value if hasattr(
                    ml.intent, "value") else ml.intent, ml.confidence
            )
        except Exception as e:
            self.logger.error("Intent classifier failed: %s", e, exc_info=True)
            ml = None

        # --------------------------------------------------------
        # RULE override check
        # rule wins unless ML strongly believes the query should
        # be semantic search (intent=SEMANTIC_SEARCH + high conf)
        # --------------------------------------------------------
        if rule_candidate:
            if (
                ml and
                ml.intent == QueryType.SEMANTIC_SEARCH and
                ml.confidence >= RULE_OVERRIDE_THRESHOLD
            ):
                context.routing_notes.append("ml_override_rule_layer")
                # fall through to ML-based semantic routing
            else:
                # Rule layer restored
                context.routing_notes.append("rule_layer_selected")
                return QueryRoute(
                    handler=rule_candidate,
                    metadata={
                        "route": "rule",
                        "ml_intent": getattr(ml.intent, "value", None) if ml else None,
                        "ml_confidence": getattr(ml, "confidence", None),
                        "ml_route_reason": "rule_not_overridden"
                    }
                )

        # -------------------------------------------------------------------
        # 3) CONFIDENCE THRESHOLD -> route to general RAG if confidence is low
        # -------------------------------------------------------------------
        CONF_THRESHOLD = 0.60
        if ml is None or ml.confidence < CONF_THRESHOLD:
            context.routing_notes.append("ml_low_confidence_to_semantic")
            sem = next(
                (h for h in self.handlers if h.__class__.__name__ ==
                 "SemanticSearchHandler"),
                None
            )
            return QueryRoute(
                handler=sem,
                metadata={
                    "route": "semantic_fallback_low_conf",
                    "ml_intent": ml.intent.value if ml else None,
                    "ml_confidence": ml.confidence if ml else None,
                    "ml_route_reason": "confidence_below_threshold"
                }
            )

        # -------------------------------------------------------------------
        # 4) Choose handler by ML intent, then let it "negotiate" via can_handle
        # -------------------------------------------------------------------
        target_handler = self.intent_to_handler.get(ml.intent)
        if target_handler is None:
            # No dedicated handler? Default to SemanticSearch.
            context.routing_notes.append("ml_handler_missing_to_semantic")
            sem = next((h for h in self.handlers if h.__class__.__name__ ==
                       "SemanticSearchHandler"), None)
            return QueryRoute(handler=sem, metadata={
                "route": "ml_missing_semantic",
                "ml_intent": ml.intent.value if hasattr(ml.intent, "value") else str(ml.intent),
                "ml_confidence": ml.confidence,
            })

        # Give the handler a chance to reject based on enriched context
        try:
            if target_handler.can_handle(context):
                context.routing_notes.append("ml_selected_handler_accepted")
                return QueryRoute(handler=target_handler, metadata={
                    "route": "ml_handler",
                    "ml_intent": ml.intent.value if hasattr(ml.intent, "value") else str(ml.intent),
                    "ml_confidence": ml.confidence,
                })
            else:
                context.routing_notes.append("ml_selected_handler_rejected")
        except Exception as e:
            self.logger.error(
                "Handler %s failed during negotiation: %s",
                target_handler.__class__.__name__, e, exc_info=True
            )
            context.routing_notes.append("ml_selected_handler_error")

        # Final fallback: SemanticSearch
        sem = next(
            (h for h in self.handlers if h.__class__.__name__ ==
             "SemanticSearchHandler"),
            None
        )
        return QueryRoute(
            handler=sem,
            metadata={
                "route": "semantic_fallback_negotiation",
                "ml_intent": ml.intent.value if ml else None,
                "ml_confidence": ml.confidence if ml else None,
                "ml_route_reason": "handler_rejected_or_missing"
            }
        )

    # =========================================================================
    # Cache + stats helpers
    # =========================================================================

    def _make_cache_key(self, context: QueryContext) -> str:
        """Deterministic cache key."""
        return f"{context.query}:{context.top_k}:{context.building_filter}"

    def _update_stats(self, handler_name: str, query_type: str, elapsed_ms: float, success: bool):
        # --- Update global totals ---
        self.stats["total_queries"] += 1
        self.stats["overall_total_ms"] += elapsed_ms

        # --- Per-handler stats ---
        hstats = self.stats["handlers"].setdefault(handler_name, {
            "count": 0,
            "total_ms": 0.0,
            "min_ms": float("inf"),
            "max_ms": 0.0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_ms": 0.0
        })

        hstats["count"] += 1
        hstats["total_ms"] += elapsed_ms
        hstats["min_ms"] = min(hstats["min_ms"], elapsed_ms)
        hstats["max_ms"] = max(hstats["max_ms"], elapsed_ms)
        if success:
            hstats["successes"] += 1

        hstats["avg_ms"] = hstats["total_ms"] / hstats["count"]
        hstats["success_rate"] = hstats["successes"] / hstats["count"]

        # --- Per query type stats ---
        tstats = self.stats["query_types"].setdefault(query_type, {
            "count": 0,
            "total_ms": 0.0,
            "min_ms": float("inf"),
            "max_ms": 0.0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_ms": 0.0
        })

        tstats["count"] += 1
        tstats["total_ms"] += elapsed_ms
        tstats["min_ms"] = min(tstats["min_ms"], elapsed_ms)
        tstats["max_ms"] = max(tstats["max_ms"], elapsed_ms)
        if success:
            tstats["successes"] += 1

        tstats["avg_ms"] = tstats["total_ms"] / tstats["count"]
        tstats["success_rate"] = tstats["successes"] / tstats["count"]

    # =========================================================================
    # Debug helpers
    # =========================================================================

    def print_handler_chain(self):
        """Display the handler chain in execution order."""
        print("\nHandler Chain (priority order):")
        for h in sorted(self.handlers, key=lambda h: h.priority):
            print(f"  {h.priority:2d}  {h.__class__.__name__}")

    def print_stats(self):
        """Print expanded telemetry for debugging."""
        print("\n=== Alfred Telemetry ===")

        total = self.stats["total_queries"]
        overall_avg = (
            self.stats["overall_total_ms"] / total if total > 0 else 0.0
        )

        print(f"Total queries: {total}")
        print(f"Overall avg time: {overall_avg:.2f} ms\n")

        print("Handlers:")
        for handler, s in self.stats["handlers"].items():
            print(f"  {handler}:")
            print(f"    Count:          {s['count']}")
            print(f"    Avg time:       {s['avg_ms']:.2f} ms")
            print(
                f"    Min/Max:        {s['min_ms']:.2f} / {s['max_ms']:.2f} ms")
            print(f"    Success rate:   {s['success_rate']:.1%}")

        print("\nQuery Types:")
        for qtype, s in self.stats["query_types"].items():
            print(f"  {qtype}:")
            print(f"    Count:          {s['count']}")
            print(f"    Avg time:       {s['avg_ms']:.2f} ms")
            print(
                f"    Min/Max:        {s['min_ms']:.2f} / {s['max_ms']:.2f} ms")
            print(f"    Success rate:   {s['success_rate']:.1%}")

        print("\n=========================\n")

    def get_statistics(self):
        total = self.stats["total_queries"]
        avg_time = (
            self.stats["overall_total_ms"] / total if total > 0 else 0.0
        )

        return {
            "total_queries": total,
            "avg_time_ms": avg_time,
            "cached_queries": self.stats["cached_queries"],
            "handlers": self.stats["handlers"],
            "query_types": self.stats["query_types"]
        }


# ============================================================================
# CONVENIENCE FUNCTION FOR EXISTING CODE
# ============================================================================


def process_query_unified(
        query: str,
        top_k: int = 10,
        **kwargs
) -> Tuple[List[Any],       # results from semantic search
           Optional[str],             # answer
           Any,             # publication_date_info
           Optional[bool]   # score_too_low
           ]:
    """
    Convenience wrapper for backward compatibility with existing code.

    Returns same format as perform_federated_search() for easy migration.

    Args:
        query: User query
        top_k: Number of results
        **kwargs: Additional context

    Returns:
        Tuple of (results, answer, publication_date_info, score_too_low)
    """
    manager = QueryManager()
    result = manager.process_query(query, top_k=top_k, **kwargs)

    return (
        result.results,
        result.answer,
        result.publication_date_info,
        result.score_too_low
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Create manager
    manager = QueryManager()

    # Example queries
    test_queries = [
        "Hello Alfred",
        "How many buildings have FRAs?",
        "Show maintenance requests for Senate House",
        "Rank buildings by area",
        "What buildings are Condition A?",
        "What is the BMS configuration for HVAC?"
    ]

    print("=" * 80)
    print("Query Manager Test Run")
    print("=" * 80)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = manager.process_query(query)
        print(f"✅ Type: {result.query_type}")
        print(f"⏱️  Time: {result.processing_time_ms:.2f}ms")
        print(f"📊 Handler: {result.handler_used}")
        print(
            f"💬 Answer preview: {result.answer[:100] if result.answer else 'No answer available'}...")

    # Show statistics
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    stats = manager.get_statistics()

    print(f"Total queries: {stats['total_queries']}")
    print(f"Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"Cached queries: {stats['cached_queries']}")

    print("\nHandlers:")
    for handler, s in stats["handlers"].items():
        print(f"  {handler}: {s['count']} uses")

    print("\nQuery Types:")
    for qtype, s in stats["query_types"].items():
        print(f"  {qtype}: {s['count']} uses")
