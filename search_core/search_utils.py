#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search utils
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any

from config import (
    TARGET_INDEXES,
    SEARCH_ALL_NAMESPACES,
    get_index_config
)
from pinecone_utils import (
    open_index,
    list_namespaces_for_index,
    vector_query,
    normalise_matches
)

from building_utils import (
    create_building_metadata_filter,
    filter_results_by_building,
    result_matches_building

)

# ============================================================================
# CONSTANTS
# ============================================================================

# Stop words for building name matching
STOP_WORDS = frozenset({
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'bms',
    'building', 'house', 'data', 'planon'
})

# Score boost for matching document types
DOC_TYPE_BOOST_FACTOR = 1.2

# Score boost for matching buildings (higher priority)
BUILDING_BOOST_FACTOR = 2.0  # 2x boost for correct building


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _namespaces_to_search(idx) -> List[Optional[str]]:
    """
    Get namespaces to search for given index.
    Returns list that may contain None for default namespace.
    """
    if not SEARCH_ALL_NAMESPACES:
        # Return list with None to indicate default namespace
        logging.debug("Using default namespace for index %s", idx)
        return [None]
    try:
        namespaces: List[Optional[str]] = list_namespaces_for_index(idx)
        # If no namespaces found, use default
        if not namespaces:
            return [None]
        return namespaces
    except RuntimeError:
        return [None]


def get_doc_type(hit: Dict[str, Any]) -> str:
    """
    Extract document type from hit consistently.
    Checks both metadata and top-level fields.
    """
    metadata = hit.get('metadata', {})
    return metadata.get('document_type') or hit.get('document_type', 'unknown')


# ============================================================================
# SEARCH OPERATIONS
# ============================================================================


def search_one_index(
    idx_name: str,
    query: str,
    k: int = 10,
    embed_model: Optional[str] = None,
    building_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search a single index with building-aware filtering.
    """
    try:
        idx = open_index(idx_name)
    except Exception as e:
        logging.warning("Failed to open index '%s': %s", idx_name, e)
        return []

    # Get index config
    index_config = get_index_config(idx_name)

    # FIXED: Ensure embed_model is always a string
    if embed_model is None:
        embed_model = index_config['model']

    # Assert that embed_model is now a string for type checker
    assert embed_model is not None, f"No embedding model configured for index {idx_name}"

    namespaces = _namespaces_to_search(idx)
    all_hits = []

    # Create building filter if specified
    metadata_filter = None
    if building_filter:
        metadata_filter = create_building_metadata_filter(building_filter)
        logging.info("🏢 Created metadata filter for building: %s",
                     building_filter)

    for ns in namespaces:
        # ADD LOGGING HERE - BEFORE THE TRY BLOCK
        if ns is None:
            logging.debug("Querying index '%s' in default namespace", idx_name)
        else:
            logging.debug("Querying index '%s' in namespace: %s", idx_name, ns)

        try:
            # Now embed_model is guaranteed to be a string
            raw = vector_query(
                idx,
                namespace=ns,
                query=query,
                k=k,
                embed_model=embed_model,  # No longer None
                metadata_filter=metadata_filter if building_filter else None
            )

            hits = normalise_matches(raw)

            # Apply building filter post-query if we have one
            if building_filter:
                hits = filter_results_by_building(hits, building_filter)

            for h in hits:
                h['index'] = idx_name
                h['namespace'] = ns

            all_hits.extend(hits)

        except Exception as e:  # pylint: disable=broad-except
            ns_display = "(default)" if ns is None else ns
            logging.warning(
                "Search failed for index='%s', namespace='%s': %s",
                idx_name, ns_display, e
            )

    return all_hits


def get_effective_score(result: Dict[str, Any]) -> float:
    """Get effective score from result (boosted or original)."""
    return result.get('boosted_score', result.get('score', 0.0))


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate results based on ID, keeping highest score.

    Args:
        results: List of search results

    Returns:
        Deduplicated results
    """
    seen_ids = {}
    for result in results:
        result_id = result.get('id')
        if not result_id:
            continue

        existing_score = get_effective_score(seen_ids.get(result_id, {}))
        current_score = get_effective_score(result)

        if result_id not in seen_ids or current_score > existing_score:
            seen_ids[result_id] = result

    return list(seen_ids.values())


def apply_doc_type_boost(
    results: List[Dict[str, Any]],
    target_doc_type: str,
    boost_factor: float = DOC_TYPE_BOOST_FACTOR
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching target document type.

    Args:
        results: Search results
        target_doc_type: Document type to boost
        boost_factor: Multiplication factor for boost

    Returns:
        Results with boosted scores
    """
    for result in results:
        doc_type = get_doc_type(result)
        if doc_type == target_doc_type:
            original_score = result.get('score', 0.0)
            result['boosted_score'] = original_score * boost_factor
            result['boost_reason'] = f'document_type:{doc_type}'

    return results


def apply_building_boost(
    results: List[Dict[str, Any]],
    target_building: str,
    boost_factor: float = BUILDING_BOOST_FACTOR
) -> List[Dict[str, Any]]:
    """
    Apply score boost to results matching target building.
    IMPROVED: Uses fuzzy matching for building comparison.

    Args:
        results: Search results
        target_building: Building name to boost
        boost_factor: Multiplication factor for boost

    Returns:
        Results with boosted scores
    """
    if not target_building:
        return results

    for result in results:
        if result_matches_building(result, target_building):
            original_score = result.get(
                'boosted_score', result.get('score', 0.0))
            result['boosted_score'] = original_score * boost_factor
            result['boost_reason'] = result.get('boost_reason', '').strip(
                ';') + f';building:{target_building}'

            # Store the matched building name
            if not result.get("building_name"):
                result['building_name'] = target_building

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def search_by_building(building_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search specifically for all documents related to a building.
    IMPROVED: Uses fuzzy matching filter.

    Args:
        building_name: Name of the building to search for
        top_k: Number of results to return

    Returns:
        List of search results for the building
    """
    all_results = []

    for idx_name in TARGET_INDEXES:
        results = search_one_index(
            idx_name,
            f"building information for {building_name}",
            top_k * 2,
            embed_model=None,
            building_filter=building_name
        )
        all_results.extend(results)

    # Filter by building using fuzzy matching
    filtered_results = filter_results_by_building(all_results, building_name)

    # Sort by effective score
    filtered_results.sort(key=get_effective_score, reverse=True)

    return filtered_results[:top_k]


def get_search_statistics(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about search results for debugging/monitoring.

    Args:
        hits: List of search results

    Returns:
        Dictionary with statistics
    """
    doc_type_counts = {}
    building_counts = {}
    index_counts = {}

    for hit in hits:
        # Document types
        doc_type = get_doc_type(hit)
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        # Buildings
        building = hit.get('building_name', 'Unknown')
        building_counts[building] = building_counts.get(building, 0) + 1

        # Indexes
        index = hit.get('index', 'Unknown')
        index_counts[index] = index_counts.get(index, 0) + 1

    return {
        'total_results': len(hits),
        'doc_types': doc_type_counts,
        'buildings': building_counts,
        'indexes': index_counts,
        'avg_score': sum(hit.get('score', 0.0) for hit in hits) / len(hits) if hits else 0,
        'max_score': max((hit.get('score', 0.0) for hit in hits), default=0),
        'min_score': min((hit.get('score', 0.0) for hit in hits), default=0),
        'avg_boosted_score': (
            sum(get_effective_score(hit) for hit in hits) / len(hits)
            if hits else 0
        ),

    }
