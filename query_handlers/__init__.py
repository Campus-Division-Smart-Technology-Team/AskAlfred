#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Handlers Package

This package contains all query type handlers that implement different
processing strategies using the Chain of Responsibility pattern.

Each handler:
- Inherits from BaseQueryHandler
- Implements can_handle() to determine if it should process a query
- Implements handle() to process the query and return a QueryResult
- Has a priority (lower number = higher priority)

Available Handlers:
- ConversationalHandler: Greetings, about, gratitude, farewells
- MaintenanceHandler: Maintenance requests and jobs
- RankingHandler: Building rankings by area/size
- PropertyHandler: Property condition queries
- CountingHandler: Counting buildings and documents
- SemanticSearchHandler: Default semantic search (fallback)

Example Usage:
    from query_handlers import ConversationalHandler
    from query_manager import QueryContext, QueryResult
    
    handler = ConversationalHandler()
    context = QueryContext(query="Hello Alfred")
    
    if handler.can_handle(context):
        result = handler.handle(context)
        print(result.answer)

Adding a New Handler:
    1. Create new file in this directory (e.g., comparison_handler.py)
    2. Inherit from BaseQueryHandler or PatternBasedHandler
    3. Implement can_handle() and handle() methods
    4. Set priority and query_type
    5. Add to QueryManager._initialize_handlers()
"""

# Base classes
from query_handlers.base_handler import (
    BaseQueryHandler,
    PatternBasedHandler
)

# Concrete handlers
from query_handlers.conversational_handler import ConversationalHandler
from query_handlers.maintenance_handler import MaintenanceHandler
from query_handlers.ranking_handler import RankingHandler
from query_handlers.property_handler import PropertyHandler
from query_handlers.counting_handler import CountingHandler
from query_handlers.semantic_search_handler import SemanticSearchHandler


# Public API
__all__ = [
    # Base classes
    'BaseQueryHandler',
    'PatternBasedHandler',

    # Handlers
    'ConversationalHandler',
    'MaintenanceHandler',
    'RankingHandler',
    'PropertyHandler',
    'CountingHandler',
    'SemanticSearchHandler',
]


# Package metadata
__version__ = '1.0.0'
__author__ = 'University of Bristol Smart Technology Team'


def get_available_handlers():
    """
    Get list of available handler classes.

    Useful for dynamic handler loading and validation.

    Returns:
        list: List of available handler classes
    """
    handlers = []

    for handler_name in __all__:
        if handler_name.endswith('Handler'):
            handler_class = globals().get(handler_name)
            if handler_class is not None:
                handlers.append(handler_class)

    return handlers


def get_handler_info():
    """
    Get information about all available handlers.

    Returns:
        dict: Dictionary mapping handler names to their info
    """
    info = {}

    for handler_class in get_available_handlers():
        try:
            # Create temporary instance to get info
            temp = handler_class()
            info[handler_class.__name__] = {
                'priority': temp.priority,
                'query_type': temp.query_type.value if temp.query_type else 'unknown',
                'class': handler_class,
                'module': handler_class.__module__,
            }
        except Exception as e:
            info[handler_class.__name__] = {
                'error': str(e),
                'class': handler_class,
            }

    return info


# Convenience function for debugging
def print_handler_chain():
    """Print the handler chain in priority order."""
    handlers = get_available_handlers()

    # Create instances and sort by priority
    instances = []
    for handler_class in handlers:
        try:
            instance = handler_class()
            instances.append(instance)
        except Exception as e:
            print(f"❌ Could not instantiate {handler_class.__name__}: {e}")

    instances.sort(key=lambda h: h.priority)

    print("=" * 80)
    print("Handler Chain (Priority Order)")
    print("=" * 80)

    for i, handler in enumerate(instances, 1):
        print(f"{i}. {handler.__class__.__name__}")
        print(f"   Priority: {handler.priority}")
        print(
            f"   Type: {handler.query_type.value if handler.query_type else 'unknown'}")
        print()


if __name__ == "__main__":
    # When run directly, show available handlers
    print("=" * 80)
    print("Query Handlers Package")
    print("=" * 80)
    print()

    available = get_available_handlers()
    print(f"Available handlers: {len(available)}")
    print()

    print_handler_chain()

    print("\nHandler Details:")
    print("-" * 80)
    for name, details in get_handler_info().items():
        print(f"\n{name}:")
        for key, value in details.items():
            if key != 'class':
                print(f"  {key}: {value}")
