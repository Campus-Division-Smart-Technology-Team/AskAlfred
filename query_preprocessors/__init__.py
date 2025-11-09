#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Preprocessors Package

This package contains preprocessors that enrich QueryContext before routing
to handlers. Preprocessors run in sequence and can extract information,
normalize text, or add metadata.

Preprocessors are optional components that run BEFORE handler routing.
They modify the QueryContext object to add useful information that handlers
can use.

Architecture:
    User Query → QueryContext → [Preprocessors] → Handler Chain → QueryResult
                                      ↑
                            Enriches context with:
                            - Building names
                            - Business terms
                            - Complexity metrics
                            - Normalized text
                            - etc.

Available Preprocessors:
- BuildingExtractor: Extracts building names from queries
- BusinessTermExtractor: Identifies and normalizes business terminology
- QueryComplexityAnalyzer: Estimates query complexity
- SpellChecker: (Optional) Corrects common misspellings

Example Usage:
    from query_preprocessors import BuildingExtractor, BusinessTermExtractor
    from query_manager import QueryContext
    
    # Create preprocessors
    building_extractor = BuildingExtractor()
    term_extractor = BusinessTermExtractor()
    
    # Create context
    context = QueryContext(query="Show FRAs for Senate House")
    
    # Run preprocessors
    building_extractor.process(context)
    term_extractor.process(context)
    
    # Context now has extracted info
    print(context.building)        # "Senate House"
    print(context.business_terms)  # [{'term': 'FRA', 'type': 'fire_risk_assessment'}]

Creating a New Preprocessor:
    1. Inherit from BasePreprocessor
    2. Implement process(context) method
    3. Add to QueryManager.__init__() preprocessors list
    
    class MyPreprocessor(BasePreprocessor):
        def process(self, context: QueryContext) -> None:
            # Extract information
            something = self.extract_from(context.query)
            
            # Add to context
            context.add_to_cache('my_data', something)
"""

# Base class
from query_preprocessors.base_preprocessor import BasePreprocessor
from query_preprocessors.building_extractor import BuildingExtractor
from query_preprocessors.business_term_extractor import BusinessTermExtractor
from query_preprocessors.query_complexity_analyser import QueryComplexityAnalyser
from query_preprocessors.spell_checker import SpellCheckPreprocessor


# Public API
__all__ = [
    # Base class
    'BasePreprocessor',

    # Preprocessors
    'BuildingExtractor',
    'BusinessTermExtractor',
    'QueryComplexityAnalyser',
    'SpellCheckPreprocessor',
]


# Package metadata
__version__ = '1.0.0'
__author__ = 'University of Bristol Smart Technology Team'


def get_available_preprocessors():
    """
    Get list of available preprocessor classes.

    Returns:
        list: List of available preprocessor classes
    """
    preprocessors = []

    for preprocessor_name in __all__:
        if preprocessor_name != 'BasePreprocessor':
            preprocessor_class = globals().get(preprocessor_name)
            if preprocessor_class is not None:
                preprocessors.append(preprocessor_class)

    return preprocessors


def get_preprocessor_info():
    """
    Get information about all available preprocessors.

    Returns:
        dict: Dictionary mapping preprocessor names to their info
    """
    info = {}

    for preprocessor_class in get_available_preprocessors():
        try:
            info[preprocessor_class.__name__] = {
                'class': preprocessor_class,
                'module': preprocessor_class.__module__,
                'docstring': preprocessor_class.__doc__.split('\n')[0] if preprocessor_class.__doc__ else 'No description'
            }
        except Exception as e:
            info[preprocessor_class.__name__] = {
                'error': str(e),
                'class': preprocessor_class,
            }

    return info


def create_default_preprocessor_pipeline():
    """
    Create default preprocessor pipeline.

    Returns:
        list: List of preprocessor instances in recommended order
    """
    pipeline = []

    # Order matters - each preprocessor can use results from previous ones
    preprocessor_classes = [
        BuildingExtractor,           # Extract building names
        BusinessTermExtractor,       # Extract business terms
        QueryComplexityAnalyser,     # Analyze complexity
        # SpellCheckPreprocessor,    # Spell check (optional, can be slow)
    ]

    for preprocessor_class in preprocessor_classes:
        if preprocessor_class is not None:
            try:
                pipeline.append(preprocessor_class())
            except Exception as e:
                print(
                    f"⚠️  Could not create {preprocessor_class.__name__}: {e}")

    return pipeline


def print_preprocessor_info():
    """Print information about available preprocessors."""
    print("=" * 80)
    print("Query Preprocessors")
    print("=" * 80)
    print()

    available = get_available_preprocessors()
    print(f"Available preprocessors: {len(available)}")
    print()

    for name, details in get_preprocessor_info().items():
        print(f"{name}:")
        if 'error' in details:
            print(f"  ❌ Error: {details['error']}")
        else:
            print(f"  📄 {details['docstring']}")
            print(f"  📦 Module: {details['module']}")
        print()


if __name__ == "__main__":
    # When run directly, show available preprocessors
    print_preprocessor_info()

    print("\nDefault Pipeline:")
    print("-" * 80)
    pipeline = create_default_preprocessor_pipeline()
    for i, preprocessor in enumerate(pipeline, 1):
        print(f"{i}. {preprocessor.__class__.__name__}")
