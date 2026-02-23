#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base handler interface for query processing.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from query_context import QueryContext
from query_result import QueryResult


class BaseQueryHandler(ABC):
    """
    Abstract base class for all query handlers.

    Each handler implements the Chain of Responsibility pattern.
    """

    def __init__(self):
        self.query_type = None
        self.priority = 0  # Lower to higher priority
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def _log_handling(self, context):
        """Log that this handler is processing query."""
        self.logger.info(
            "Handling query: '%s...' (type: %s)",
            context.query[:50],
            self.query_type.value if self.query_type else 'unknown'
        )

    @abstractmethod
    def can_handle(self, context: QueryContext) -> bool:
        """
        Determine if this handler can process the query.

        Args:
            context: Query context with all information

        Returns:
            True if handler can process this query
        """
        pass

    @abstractmethod
    def handle(self, context: QueryContext) -> QueryResult:
        """
        Process the query and generate result.

        Args:
            context: Query context

        Returns:
            QueryResult with answer and metadata
        """
        pass

    def get_confidence(self, context: QueryContext) -> float:
        """
        Return confidence score for handling this query.

        Override for more nuanced routing decisions.

        Returns:
            Confidence score 0.0-1.0
        """
        return 1.0 if self.can_handle(context) else 0.0

    def get_metadata(self, context: QueryContext) -> Dict[str, Any]:
        """
        Extract handler-specific metadata from context.

        Override to provide routing metadata.
        """
        return {}


class PatternBasedHandler(BaseQueryHandler):
    """
    Base class for handlers that use regex patterns for matching.

    Subclasses just need to define self.patterns and implement handle().
    """

    def __init__(self):
        super().__init__()
        self.patterns = []  # List of compiled regex patterns

    def can_handle(self, context) -> bool:
        """Check if query matches any pattern."""
        query = context.query.lower().strip()

        for pattern in self.patterns:
            if pattern.search(query):
                # Cache matched pattern for use in handle()
                context.add_to_cache('matched_pattern', pattern)
                return True

        return False

    def get_metadata(self, context) -> Dict[str, Any]:
        """Return matched pattern info."""
        matched = context.get_from_cache('matched_pattern')
        if matched:
            return {'matched_pattern': matched.pattern}
        return {}
