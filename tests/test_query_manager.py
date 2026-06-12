"""
Tests for the Query Manager,
ensuring correct routing of queries to handlers and backward compatibility of results.
"""

import pytest

from query_core.query_manager import QueryManager, process_query_unified
from query_core.query_result import QueryResult
from query_core.query_types import QueryType

# Test queries with expected routing
TEST_CASES = [
    # (query, expected_query_type)
    ("Hello Alfred", QueryType.CONVERSATIONAL),
    ("Who are you?", QueryType.CONVERSATIONAL),
    ("Thank you", QueryType.CONVERSATIONAL),
    ("Goodbye", QueryType.CONVERSATIONAL),
    ("How many buildings have FRAs?", QueryType.COUNTING),
    ("Count the buildings", QueryType.COUNTING),
    ("Number of buildings with BMS", QueryType.COUNTING),
    ("Show maintenance requests for Senate House", QueryType.MAINTENANCE),
    ("List all maintenance jobs", QueryType.MAINTENANCE),
    ("Electrical maintenance requests", QueryType.MAINTENANCE),
    ("Rank buildings by area", QueryType.RANKING),
    ("Top 10 largest buildings", QueryType.RANKING),
    ("Sort buildings by gross area", QueryType.RANKING),
    ("Which buildings are Condition A?", QueryType.PROPERTY_CONDITION),
    ("Show derelict buildings", QueryType.PROPERTY_CONDITION),
    ("What is the BMS configuration?", QueryType.SEMANTIC_SEARCH),
    ("Tell me about HVAC systems", QueryType.SEMANTIC_SEARCH),
]


class TestQueryManager:
    """Test the Query Manager."""

    def setup_method(self):
        """Setup before each test."""
        self.manager = QueryManager()

    @pytest.mark.parametrize("query,expected_type", TEST_CASES)
    def test_query_routing(self, query, expected_type, monkeypatch):
        """Test that queries route to correct handlers."""

        # Routing tests must not execute the live maintenance path (Pinecone).
        def fake_maintenance_handle(self, context):
            return QueryResult(
                query=context.query,
                answer="maintenance response",
                handler_used="MaintenanceHandler",
                query_type=QueryType.MAINTENANCE.value,
            )

        monkeypatch.setattr(
            "query_handlers.maintenance_handler.MaintenanceHandler.handle",
            fake_maintenance_handle,
        )

        result = self.manager.process_query(query)

        assert result.query_type == expected_type.value, (
            f"Query '{query}' routed to {result.query_type}, "
            f"expected {expected_type.value}"
        )
        assert result.success, f"Query failed: {result.metadata.get('error')}"
        assert (
            result.answer is not None and len(result.answer) > 0
        ), "Empty answer returned"

    def test_conversational_responses(self):
        """Test conversational handler returns appropriate responses."""
        greetings = ["hello", "hi", "hey Alfred"]

        for greeting in greetings:
            result = self.manager.process_query(greeting)
            assert result.answer is not None and "Alfred" in result.answer
            assert result.answer is not None and "help" in result.answer.lower()

    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Empty query
        result = self.manager.process_query("")
        # Should still return a result (even if error)
        assert result is not None
        assert isinstance(result.answer, str)

    def test_statistics(self):
        """Test statistics tracking."""
        # Process some queries
        queries = ["hello", "how many buildings?", "rank buildings by area"]

        for query in queries:
            self.manager.process_query(query)

        # Check stats
        stats = self.manager.get_statistics()
        assert stats["total_queries"] == len(queries)
        assert len(stats["query_types"]) > 0
        assert stats["avg_time_ms"] > 0

    def test_cache_entries_do_not_share_mutable_state(self):
        """Stored and returned cache values are independent deep copies."""
        self.manager.cache_enabled = True
        original = QueryResult(
            query="hello",
            answer="hi",
            results=[{"items": [1]}],
            metadata={"nested": {"value": 1}},
        )

        self.manager._store_cached_result("cache-key", original)
        original.results[0]["items"].append(2)
        original.metadata["nested"]["value"] = 2

        first = self.manager._get_cached_result("cache-key")
        assert first is not None
        assert first.results == [{"items": [1]}]
        assert first.metadata == {"nested": {"value": 1}}

        first.results[0]["items"].append(3)
        first.metadata["nested"]["value"] = 3

        second = self.manager._get_cached_result("cache-key")
        assert second is not None
        assert second.results == [{"items": [1]}]
        assert second.metadata == {"nested": {"value": 1}}


class TestBackwardCompatibility:
    """Test that results match old system format."""

    def test_result_format(self):
        """Test QueryResult has all expected fields."""

        query = "What is the BMS configuration?"
        results, answer, pub_date, score_low = process_query_unified(query)

        assert isinstance(results, list)
        assert isinstance(answer, str)
        assert isinstance(pub_date, str)
        assert isinstance(score_low, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
