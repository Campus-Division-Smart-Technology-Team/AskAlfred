#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Query Manager
"""

import pytest
from query_types import QueryType
from query_manager import QueryManager


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
    def test_query_routing(self, query, expected_type):
        """Test that queries route to correct handlers."""
        result = self.manager.process_query(query)

        assert result.query_type == expected_type.value, (
            f"Query '{query}' routed to {result.query_type}, "
            f"expected {expected_type.value}"
        )
        assert result.success, f"Query failed: {result.metadata.get('error')}"
        assert len(result.answer) > 0, "Empty answer returned"

    def test_conversational_responses(self):
        """Test conversational handler returns appropriate responses."""
        greetings = ["hello", "hi", "hey Alfred"]

        for greeting in greetings:
            result = self.manager.process_query(greeting)
            assert "Alfred" in result.answer
            assert "help" in result.answer.lower()

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
        queries = [
            "hello",
            "how many buildings?",
            "rank buildings by area"
        ]

        for query in queries:
            self.manager.process_query(query)

        # Check stats
        stats = self.manager.get_statistics()
        assert stats['total_queries'] == len(queries)
        assert len(stats['by_type']) > 0
        assert stats['avg_time_ms'] > 0


class TestBackwardCompatibility:
    """Test that results match old system format."""

    def test_result_format(self):
        """Test QueryResult has all expected fields."""
        from query_manager import process_query_unified

        query = "What is the BMS configuration?"
        results, answer, pub_date, score_low = process_query_unified(query)

        assert isinstance(results, list)
        assert isinstance(answer, str)
        assert isinstance(pub_date, str)
        assert isinstance(score_low, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
