#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building name extraction and lookup utilities with dynamic Pinecone index support.
IMPROVED VERSION: Enhanced fuzzy matching against multiple metadata fields.
FIXED: Now works with lowercase queries like "does old park hill have an fra?"

NOTE:
+   This is the *authoritative* building name extractor for AskAlfred.
+   All other modules must call `extract_building_from_query` instead of
+   implementing local regex or fuzzy logic.

Key improvements:
- Fuzzy matching (80% threshold) against multiple metadata fields
- Support for Property names, UsrFRACondensedPropertyName, building_name, canonical_building_name
- Enhanced metadata search strategy for better building detection
- Better handling of variations and aliases
- FIXED: Regex patterns now accept lowercase input
- ADDED: N-gram fallback for queries that don't match patterns
"""

import re
import json
from typing import Optional, Any
from difflib import get_close_matches, SequenceMatcher
import logging
import threading
from .normaliser import normalise_building_name
from .cache import _CACHE_STATE
from pinecone_utils import open_index
from config import (
    get_index_config,
    BUILDING_UTILS_FUZZY_MATCH_THRESHOLD,
    BUILDING_UTILS_MIN_NAME_LENGTH,
    TARGET_INDEXES,
)
from emojis import (EMOJI_CROSS,
                    EMOJI_BUILDING, EMOJI_CAUTION, EMOJI_TICK, EMOJI_SEARCH)
# ============================================================================
# BUILDING NAME CACHE (populated from Pinecone at startup)
# ============================================================================

from .cache import (
    BUILDING_NAMES_CACHE as _BUILDING_NAMES_CACHE,
    BUILDING_ALIASES_CACHE as _BUILDING_ALIASES_CACHE,
    METADATA_FIELDS_CACHE as _METADATA_FIELDS_CACHE,
    INDEXES_WITH_BUILDINGS as _INDEXES_WITH_BUILDINGS,
)
from .validation import INVALID_BUILDING_NAMES, is_valid_building_name


class BuildingCacheManager:
    """
    Manages building name cache with lazy initialisation.
    Thread-safe singleton pattern for cache population.
    """
    _initialised = False
    _lock = threading.Lock()
    _initialisation_attempted = False

    @classmethod
    def ensure_initialised(cls, force: bool = False) -> bool:
        """
        Ensure building cache is populated. Thread-safe lazy initialisation.

        Args:
            force: If True, reinitialise even if already done

        Returns:
            True if cache is populated, False otherwise
        """
        if cls._initialised and not force:
            return True

        if cls._initialisation_attempted and not force:
            return cls._initialised

        with cls._lock:
            # Double-check locking pattern
            if cls._initialised and not force:
                return True

            cls._initialisation_attempted = True

            try:
                if _CACHE_STATE["populated"] and not force:
                    cls._initialised = True
                    return True

                logging.info("Initialising building cache...")
                results = populate_building_cache_from_multiple_indexes(
                    TARGET_INDEXES)

                total_buildings = sum(results.values())
                if total_buildings > 0:
                    cls._initialised = True
                    logging.info(
                        "%s Building cache initialised: %s buildings", EMOJI_TICK, total_buildings)
                    return True
                else:
                    logging.warning(
                        "%s Building cache initialisation found no buildings", EMOJI_CAUTION)
                    return False

            except Exception as e:
                logging.error(
                    "%s Failed to initialise building cache: %s", EMOJI_CROSS, e, exc_info=True)
                return False

    @classmethod
    def get_known_buildings(cls) -> list[str]:
        """Get list of known buildings, initializing cache if needed."""
        cls.ensure_initialised()
        return list(set(_BUILDING_NAMES_CACHE.values())) if _CACHE_STATE["populated"] else []

    @classmethod
    def get_alias(cls, canonical_or_alias: str) -> Optional[str]:
        """Get canonical name for an alias."""
        cls.ensure_initialised()
        return _BUILDING_ALIASES_CACHE.get(canonical_or_alias.lower()) if _CACHE_STATE["populated"] else None

    @classmethod
    def is_populated(cls) -> bool:
        """Check if cache is populated."""
        return _CACHE_STATE["populated"]

    @classmethod
    def set_populated(cls, value: bool):
        """Set cache populated flag."""
        _CACHE_STATE["populated"] = value
        cls._initialised = value

    @classmethod
    def get_cache_stats(cls) -> dict[str, Any]:
        """Get statistics about the cache."""
        return {
            'initialised': cls._initialised,
            'canonical_names': len(set(_BUILDING_NAMES_CACHE.values())),
            'aliases': len(_BUILDING_ALIASES_CACHE),
            'metadata_fields': sum(len(v) for v in _METADATA_FIELDS_CACHE.values()),
            'indexes_with_buildings': _INDEXES_WITH_BUILDINGS.copy()
        }


# Metadata fields to search for building names (in priority order)
BUILDING_METADATA_FIELDS = [
    'canonical_building_name',
    'building_name',
    'Property names',
    'UsrFRACondensedPropertyName',
    'building_aliases'
]

# Fuzzy match threshold (80% similarity)
FUZZY_MATCH_THRESHOLD = BUILDING_UTILS_FUZZY_MATCH_THRESHOLD

# ============================================================================
# BUILDING PATTERNS (for query extraction) - FIXED FOR LOWERCASE
# ============================================================================

# Common building name patterns (IMPROVED: now accepts lowercase input)
BUILDING_PATTERNS = [
    # Pattern 1: Allow optional number prefix like "1-9" or "123"
    re.compile(
        r'\bat\s+((?:\d+[\s\-]*)*[A-Za-z][A-Za-z0-9\s\-\']+)', re.IGNORECASE),

    # Pattern 2: "in <building>" - accepts lowercase
    re.compile(
        r'\bin\s+((?:\d+[\s\-]*)*[A-Za-z][A-Za-z\s\-\']+)', re.IGNORECASE),

    # Pattern 3: "for <building>" - accepts lowercase
    re.compile(
        r'\bfor\s+((?:\d+[\s\-]*)*[A-Za-z][A-Za-z\s\-\']+)', re.IGNORECASE),

    # Pattern 4: "of <building>" - NEW: for "FRA of building" queries
    re.compile(
        r'\bof\s+((?:\d+[\s\-]*)*[A-Za-z][A-Za-z\s\-\']+)', re.IGNORECASE),

    # Pattern 5: "<building> building/house/etc" - accepts lowercase
    re.compile(
        r'\b([A-Za-z][A-Za-z\s\-\']+)\s+(?:Building|House|Hall|Centre|Center|Complex|Tower)\b', re.IGNORECASE),

    # Pattern 6: “at X”, "in X", "for X", "of X" (limit trailing tokens)
    re.compile(
        r"\b(?:at|in|for|of)\s+([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+){0,3})", re.IGNORECASE),

    # Pattern 7: Proper-noun sequences (2–4 words) but disallow stopwords
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"),

    # Pattern 8: "tell me about <building>" (1–4 words)
    re.compile(
        r"\btell\s+me\s+about\s+([A-Za-z][A-Za-z0-9\s\-\']{2,})",
        re.IGNORECASE)
]

# Question words and common words to filter out
QUESTION_WORDS = frozenset({
    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'tell', 'show',
    'find', 'search', 'get', 'give', 'list', 'are', 'is', 'do', 'does',
    'can', 'could', 'would', 'should', 'fire', 'risk', 'assessment',
    'the', 'a', 'an', 'have', 'has', 'there', 'their', 'this', 'that',
    'these', 'those', 'my', 'our', 'your', 'its', 'fra', 'fras',
    'maintenance', 'request', 'requests', 'job', 'jobs'
})

# Minimum length for building names
MIN_BUILDING_NAME_LENGTH = BUILDING_UTILS_MIN_NAME_LENGTH


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================


def populate_building_cache_from_multiple_indexes(
    index_names: list[str],
    namespace: Optional[str] = None
) -> dict[str, int]:
    """
    Populate building name cache from multiple Pinecone indexes.
    Tries all indexes and aggregates results.

    Args:
        index_names: List of index names to try
        namespace: Namespace to query

    Returns:
        Dictionary mapping index names to number of buildings found
    """
    # Clear existing cache
    _BUILDING_NAMES_CACHE.clear()
    _BUILDING_ALIASES_CACHE.clear()
    _METADATA_FIELDS_CACHE.clear()
    _INDEXES_WITH_BUILDINGS.clear()

    results = {}
    total_buildings = 0

    # Import here to avoid circular dependency
    # from pinecone_utils import open_index

    for idx_name in index_names:
        try:
            logging.info("Trying to populate cache from index '%s'", idx_name)
            idx = open_index(idx_name)
            buildings_found = populate_building_cache_from_index(
                idx, namespace, index_name=idx_name, skip_if_populated=False
            )

            results[idx_name] = buildings_found
            total_buildings += buildings_found

            if idx_name not in _INDEXES_WITH_BUILDINGS and buildings_found > 0:
                _INDEXES_WITH_BUILDINGS.append(idx_name)
                logging.info("%s Index '%s' has building data (%d buildings)",
                             EMOJI_BUILDING, idx_name, buildings_found)
            else:
                logging.info(
                    "%s Index '%s' has no building data", EMOJI_CROSS, idx_name)

        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Failed to check index '%s': %s", idx_name, e)
            results[idx_name] = 0

    if total_buildings > 0:
        BuildingCacheManager.set_populated(True)
        logging.info(
            "%s Building cache initialised from %d/%d indexes: %d total buildings", EMOJI_TICK,
            len(_INDEXES_WITH_BUILDINGS),
            len(index_names),
            total_buildings
        )
    else:
        logging.warning(
            "%s No building data found in any of %d indexes", EMOJI_CROSS,
            len(index_names)
        )
        BuildingCacheManager.set_populated(False)

    return results


def resolve_building_name_fuzzy(raw_name: Optional[str]) -> Optional[str]:
    """
    Resolve any raw building name to a canonical building name
    using fuzzy matching + alias lookup + normalisation.

    This consolidates logic currently scattered inside search_operations.
    """
    if not raw_name:
        return None

    name = raw_name.strip().lower()

    # 1. Exact alias match
    if name in _BUILDING_ALIASES_CACHE:
        return _BUILDING_ALIASES_CACHE[name]

    # 2. Exact canonical match
    if name in _BUILDING_NAMES_CACHE:
        return _BUILDING_NAMES_CACHE[name]

    # 3. Fuzzy match
    fuzzy_match = validate_building_name_fuzzy(name)
    if fuzzy_match:
        norm = normalise_building_name(fuzzy_match)
        if norm.lower() in _BUILDING_NAMES_CACHE:
            return _BUILDING_NAMES_CACHE[norm.lower()]
        return norm

    # 4. As-is fallback
    return raw_name


def create_building_metadata_filter(building_filter: str) -> Optional[dict[str, Any]]:
    """
    Create comprehensive Pinecone metadata filter for building matching.
    Includes fuzzy matching conditions for all metadata fields.

    Args:
        building_filter: Building name to filter by

    Returns:
        Pinecone filter dictionary or None if no conditions created
    """
    canonical = _BUILDING_NAMES_CACHE.get(
        building_filter.lower(), building_filter)
    normalised_building = normalise_building_name(canonical)

    filter_conditions = []

    # Add exact matches for all metadata fields
    for field in BUILDING_METADATA_FIELDS:
        filter_conditions.append({field: {"$eq": building_filter}})
        filter_conditions.append({field: {"$eq": normalised_building}})
        filter_conditions.append({field: {"$eq": building_filter.lower()}})
        filter_conditions.append({field: {"$eq": building_filter.upper()}})

    # Add conditions for known aliases if cache is populated
    if _CACHE_STATE["populated"] and _BUILDING_ALIASES_CACHE:
        # Find all aliases that map to this building
        for alias, canonical in _BUILDING_ALIASES_CACHE.items():
            if canonical == building_filter or canonical.lower() == building_filter.lower():
                for field in BUILDING_METADATA_FIELDS:
                    filter_conditions.append({field: {"$eq": alias}})
                    filter_conditions.append({field: {"$eq": alias.title()}})

    # Add conditions for all known metadata field variations
    canonical = _BUILDING_NAMES_CACHE.get(
        building_filter.lower(), building_filter)
    if canonical in _METADATA_FIELDS_CACHE:
        for variation in _METADATA_FIELDS_CACHE[canonical]:
            for field in BUILDING_METADATA_FIELDS:
                filter_conditions.append({field: {"$eq": variation}})
                filter_conditions.append({field: {"$eq": variation.lower()}})
                filter_conditions.append({field: {"$eq": variation.title()}})
                filter_conditions.append({field: {"$eq": variation.upper()}})

    # Remove duplicates while preserving order
    seen = set()
    unique_conditions = []
    for condition in filter_conditions:
        # Convert dict to string for comparison
        condition_str = str(sorted(condition.items()))
        if condition_str not in seen:
            seen.add(condition_str)
            unique_conditions.append(condition)

    return {"$or": unique_conditions} if unique_conditions else None


def resolve_building_alias(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    lower = name.lower()

    # If direct alias exists → return canonical
    if lower in _BUILDING_ALIASES_CACHE:
        return _BUILDING_ALIASES_CACHE[lower]

    # If already canonical, return it
    if lower in _BUILDING_NAMES_CACHE:
        return _BUILDING_NAMES_CACHE[lower]

    # Nothing found
    return name


def populate_building_cache_from_index(
    idx: Any,
    namespace: Optional[str] = None,
    index_name: Optional[str] = None,
    skip_if_populated: bool = True
) -> int:
    """
    Populate building name cache from Pinecone index metadata.
    IMPROVED: Extracts data from multiple metadata fields.

    Args:
        idx: Pinecone index object
        namespace: Namespace to query
        index_name: Name of the index (optional)
        skip_if_populated: If True, skip if cache already populated

    Returns:
        Number of buildings found in this index
    """
    if skip_if_populated and BuildingCacheManager.is_populated():
        logging.info("Building cache already populated, skipping")
        return len(set(_BUILDING_NAMES_CACHE.values()))

    logging.info("Attempting to populate building cache from index '%s'...",
                 index_name or "unknown")

    try:
        # Import config here to avoid circular imports
        # from config import get_index_config

        # Determine the index name
        resolved_index_name: str = index_name or 'local-docs'

        # Get the correct dimension for this index
        index_config = get_index_config(resolved_index_name)
        dimension = index_config['dimension']
        logging.info("Using dimension %d for index '%s'",
                     dimension, resolved_index_name)

        # Query for Planon data records
        # Try multiple namespaces since planon_data might be in planon_data namespace
        dummy_vector = [0.0] * dimension

        # List of namespaces to try
        namespaces_to_try = [namespace, "planon_data", None]
        matches = []

        for ns in namespaces_to_try:
            try:
                results = idx.query(
                    vector=dummy_vector,
                    filter={"document_type": {"$eq": "planon_data"}},
                    top_k=1000,  # Adjust based on number of buildings
                    namespace=ns,
                    include_metadata=True
                )

                current_matches = results.get("matches", [])
                if current_matches:
                    logging.info("Found %d planon_data records in namespace '%s'",
                                 len(current_matches), ns or "default")
                    matches.extend(current_matches)
            except Exception as e:
                logging.debug("Error querying namespace '%s': %s", ns, e)
                continue

        if not matches:
            logging.info("%s No planon_data found in index '%s' (tried namespaces: %s)", EMOJI_CROSS,
                         resolved_index_name,
                         [ns or "default" for ns in namespaces_to_try])
            return 0

        canonical_names = set()
        new_names_count = 0
        new_aliases_count = 0
        new_metadata_fields_count = 0

        for match in matches:
            metadata = match.get("metadata", {})

            # Extract canonical name (priority order)
            canonical = None
            for field in ['canonical_building_name', 'building_name']:
                if metadata.get(field):
                    canonical = metadata[field]
                    break

            if not canonical:
                continue

            canonical_names.add(canonical)

            # Initialise metadata fields set for this building
            if canonical not in _METADATA_FIELDS_CACHE:
                _METADATA_FIELDS_CACHE[canonical] = set()

            # Map normalised canonical name
            normalised = canonical.lower().strip()
            if normalised not in _BUILDING_NAMES_CACHE:
                _BUILDING_NAMES_CACHE[normalised] = canonical
                new_names_count += 1

            # Add canonical to metadata fields
            _METADATA_FIELDS_CACHE[canonical].add(canonical)
            _METADATA_FIELDS_CACHE[canonical].add(normalised)

            # Extract and store ALL variations from metadata fields
            for field in BUILDING_METADATA_FIELDS:
                field_value = metadata.get(field)

                if not field_value:
                    continue

                # Handle list values (e.g. Property names)
                if isinstance(field_value, list):
                    for item in field_value:
                        if item and str(item).strip():
                            value = str(item).strip()
                            _METADATA_FIELDS_CACHE[canonical].add(value)
                            _METADATA_FIELDS_CACHE[canonical].add(
                                value.lower())
                            new_metadata_fields_count += 1

                # Handle string values
                elif isinstance(field_value, str) and field_value.strip():
                    value = field_value.strip()
                    _METADATA_FIELDS_CACHE[canonical].add(value)
                    _METADATA_FIELDS_CACHE[canonical].add(value.lower())
                    new_metadata_fields_count += 1

            # Extract building aliases if present
            aliases_field = metadata.get('building_aliases') or metadata.get(
                'aliases')
            if aliases_field:
                aliases = []
                if isinstance(aliases_field, str):
                    # Try JSON list first
                    try:
                        parsed = json.loads(aliases_field)
                        if isinstance(parsed, list):
                            aliases = parsed
                        else:
                            aliases = [aliases_field]
                    except json.JSONDecodeError:
                        # Fallback: split comma-separated alias string
                        aliases = [a.strip() for a in aliases_field.split(',')]

                elif isinstance(aliases_field, list):
                    aliases = aliases_field

                for alias in aliases:
                    if alias and str(alias).strip():
                        alias_str = str(alias).strip()
                        # Extract inner aliases from bracket groups e.g. "[retort house, the sheds, bdfi]"
                        if '[' in alias_str and ']' in alias_str:
                            inside = alias_str.split(
                                '[', 1)[1].split(']', 1)[0]
                            for a in inside.split(','):
                                a_clean = a.strip().lower()
                                if a_clean and a_clean not in _BUILDING_ALIASES_CACHE:
                                    _BUILDING_ALIASES_CACHE[a_clean] = canonical
                                    new_aliases_count += 1  # count these too

                        alias_lower = alias_str.lower()

                        # Store full alias form
                        if alias_lower not in _BUILDING_ALIASES_CACHE:
                            _BUILDING_ALIASES_CACHE[alias_lower] = canonical
                            new_aliases_count += 1

                        # Always tokenize, not only when full alias is new
                        tokens = re.split(r"[,\[\]]", alias_lower)
                        for token in tokens:
                            token = token.strip().replace(" ", "").replace(".", "")
                            if token and token not in _BUILDING_ALIASES_CACHE:
                                _BUILDING_ALIASES_CACHE[token] = canonical
                                new_aliases_count += 1
                            # NEW: also split alias into tokens to catch standalone short aliases like "bdfi"
                            for token in re.split(r"[,\[\]]", alias_lower):
                                token = token.strip()
                                token = token.replace(" ", "").replace(".", "")
                                if token and token not in _BUILDING_ALIASES_CACHE:
                                    _BUILDING_ALIASES_CACHE[token] = canonical

                        # Also add to metadata fields
                        _METADATA_FIELDS_CACHE[canonical].add(alias_str)
                        _METADATA_FIELDS_CACHE[canonical].add(alias_lower)

        num_canonical = len(canonical_names)

        if num_canonical > 0:
            BuildingCacheManager.set_populated(True)
            logging.info(
                "%s Populated cache from index '%s': %d canonical buildings, "
                "%d new canonical entries, %d new aliases, %d metadata field variations",
                EMOJI_TICK,
                resolved_index_name,
                num_canonical,
                new_names_count,
                new_aliases_count,
                new_metadata_fields_count
            )
        return num_canonical

    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Error populating building cache from index '%s': %s",
            index_name or "unknown",
            e,
            exc_info=True
        )
        return 0


def clear_building_cache():
    """Clear the building name cache."""
    _BUILDING_NAMES_CACHE.clear()
    _BUILDING_ALIASES_CACHE.clear()
    _METADATA_FIELDS_CACHE.clear()
    BuildingCacheManager.set_populated(False)
    _INDEXES_WITH_BUILDINGS.clear()
    logging.info("Building cache cleared")


def get_cache_status() -> dict[str, Any]:
    """
    Get current cache status.

    Returns:
        Dictionary with cache statistics
    """
    return {
        'populated': BuildingCacheManager.is_populated(),
        'canonical_names': len(set(_BUILDING_NAMES_CACHE.values())),
        'aliases': len(_BUILDING_ALIASES_CACHE),
        'metadata_fields': sum(len(v) for v in _METADATA_FIELDS_CACHE.values()),
        'indexes_with_buildings': _INDEXES_WITH_BUILDINGS.copy()
    }


def get_building_names_from_cache() -> list[str]:
    """
    Get list of all canonical building names from cache.

    Returns:
        List of canonical building names
    """
    if not BuildingCacheManager.is_populated():
        return []
    return sorted(list(set(_BUILDING_NAMES_CACHE.values())))


def get_indexes_with_buildings() -> list[str]:
    """
    Get list of indexes that contain building data.

    Returns:
        List of index names
    """
    return _INDEXES_WITH_BUILDINGS.copy()


# ============================================================================
# FUZZY MATCHING UTILITIES
# ============================================================================


def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate fuzzy match score between two strings.
    Used by search_operations.py for building matching.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_fuzzy_matches(
    query: str,
    candidates: list[str],
    threshold: float = FUZZY_MATCH_THRESHOLD
) -> list[tuple[str, float]]:
    """
    Find fuzzy matches with scores above threshold.

    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum similarity threshold

    Returns:
        List of (match, score) tuples, sorted by score descending
    """
    query_lower = query.lower().strip()
    matches = []

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()

        if ratio >= threshold:
            matches.append((candidate, ratio))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def fuzzy_match_against_metadata_fields(
    query: str,
    canonical_name: str,
    threshold: float = FUZZY_MATCH_THRESHOLD
) -> Optional[tuple[str, float]]:
    """
    Fuzzy match query against all metadata field variations for a building.

    Args:
        query: Query string to match
        canonical_name: Canonical building name
        threshold: Minimum similarity threshold

    Returns:
        (matched_field, score) tuple or None if no match
    """
    if canonical_name not in _METADATA_FIELDS_CACHE:
        return None

    metadata_fields = _METADATA_FIELDS_CACHE[canonical_name]
    matches = find_fuzzy_matches(query, list(metadata_fields), threshold)

    if matches:
        return matches[0]  # Return best match
    return None

# ============================================================================
# BUILDING EXTRACTION FROM QUERY - IMPROVED WITH N-GRAM FALLBACK
# ============================================================================


def extract_building_from_query_ngram_fallback(
    query: str,
    known_buildings: list[str]
) -> Optional[str]:
    """
    Fallback extraction method using n-gram matching.
    Tries all 2-4 word combinations and validates against known buildings.

    This works well with lowercase input where pattern matching fails.

    Args:
        query: User query string
        known_buildings: List of known building names from cache

    Returns:
        Canonical building name if found, None otherwise
    """
    if not query or not known_buildings:
        return None

    query_lower = query.lower()
    words = query_lower.split()

    # Try longer n-grams first (4-grams, then 3-grams, then 2-grams)
    for n in range(min(4, len(words)), 1, -1):
        for i in range(len(words) - n + 1):
            candidate_words = words[i:i+n]

            # Skip if contains too many question words
            question_word_count = sum(
                1 for w in candidate_words if w in QUESTION_WORDS
            )
            if question_word_count > len(candidate_words) / 2:
                continue

            candidate = ' '.join(candidate_words)

            # Must be at least MIN_BUILDING_NAME_LENGTH characters
            if len(candidate) < MIN_BUILDING_NAME_LENGTH:
                continue

            # Validate against known buildings
            validated = validate_building_name_fuzzy(
                candidate, known_buildings)
            if validated:
                logging.info(
                    "%s N-gram fallback matched: '%s' -> '%s'",
                    EMOJI_TICK,
                    candidate,
                    validated
                )
                return validated

    return None


def extract_building_from_query(
    query: str,
    known_buildings: Optional[list[str]] = None,
    use_cache: bool = True
) -> Optional[str]:
    """
    Extract building name from user query using multiple strategies.
    Works with lowercase input and uses n-gram fallback.

    Args:
        query: User query string (works with lowercase)
        known_buildings: Optional list of known building names
        use_cache: Whether to use cached building data

    Returns:
        Canonical building name if found, None otherwise
    """
    # Validation
    if not query or not query.strip():
        return None
    logging.info(
        "%s EXTRACT: cache_populated = {BuildingCacheManager.is_populated()}, query = '%s'", EMOJI_SEARCH, query)

    # Try cache if enabled
    if use_cache and BuildingCacheManager.is_populated():
        known_buildings = get_building_names_from_cache()
    elif use_cache and not BuildingCacheManager.is_populated():
        logging.debug("Building cache not populated, extraction limited")
        return None

    if not known_buildings:
        return None

    # Reject obvious non-building maintenance keywords early
    ql = query.lower()
    query_words = set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", ql))

    # Reject queries containing maintenance keywords *only when no building name present*
    if any(word in query_words for word in INVALID_BUILDING_NAMES):
        has_building_words = False
        if BuildingCacheManager.is_populated():
            for alias_lower, canonical in _BUILDING_ALIASES_CACHE.items():
                if alias_lower in query_words:
                    has_building_words = True
                    logging.debug(
                        "%s Found building alias in query: '%s' -> '%s'",
                        EMOJI_TICK,
                        alias_lower,
                        canonical
                    )
                    break
        # Check canonical building name words if no alias found
        if not has_building_words:
            for building in (known_buildings or []):
                building_lower = building.lower().strip()
                if building_lower in INVALID_BUILDING_NAMES:
                    # 👈 don't let "Maintenance" rescue the query
                    continue

                building_words = [
                    w.lower()
                    for w in building.split()
                    if len(w) > 2
                    and not w.isdigit()
                    and w.replace('-', '').replace('/', '').isalnum()
                ]

                if len(building_words) >= 2:
                    matches = sum(
                        1 for w in building_words if w in query_words)
                    if matches >= 2:
                        has_building_words = True
                        logging.debug(
                            "%s Found building words: %s words from '%s' in query",
                            EMOJI_TICK, matches, building
                        )
                        break
                elif len(building_words) == 1 and building_words[0] in query_words:
                    has_building_words = True
                    logging.debug(
                        "%s Found building word: '%s' from '%s' in query",
                        EMOJI_TICK, building_words[0], building
                    )
                    break

        if not has_building_words:
            logging.debug(
                "%s Query contains maintenance keywords and no building words — not a building request",
                EMOJI_CROSS,
            )
            return None

    # Phase 1: Try pattern matching with improved filtering
    extracted_name = None

    for pattern in BUILDING_PATTERNS:
        matches = pattern.findall(query)
        if matches:
            # Get the first match
            candidate = matches[0] if isinstance(
                matches[0], str) else matches[0][0]
            candidate = candidate.strip()

            # IMPROVED: Clean up candidate by removing trailing question words
            words = candidate.split()
            cleaned_words = []
            for idx, word in enumerate(words):
                word_lower = word.lower()

                # allow question words ONLY if they are at the beginning
                if word_lower in QUESTION_WORDS and idx > 0:
                    logging.info(
                        "%s I rejected '%s'", EMOJI_CROSS, word_lower)
                    break

                # skip leading question words entirely
                if idx == 0 and word_lower in QUESTION_WORDS:
                    logging.info(
                        "%s  Skipping '%s'", EMOJI_CAUTION, word_lower)
                    continue

                cleaned_words.append(word)

            if cleaned_words:
                candidate = ' '.join(cleaned_words)

                # Filter: must be at least MIN_BUILDING_NAME_LENGTH chars
                if (len(candidate) >= MIN_BUILDING_NAME_LENGTH and
                        candidate.lower() not in QUESTION_WORDS):
                    extracted_name = candidate
                    logging.debug("%s Pattern extracted: '%s'", EMOJI_TICK,
                                  extracted_name)
                    break

    # If pattern matching succeeded, validate it
    if extracted_name:
        logging.debug("Extracted potential building name: '%s'",
                      extracted_name)
        validated = validate_building_name_fuzzy(
            extracted_name, known_buildings)
        if validated:
            return validated

        # If validation failed, fall through to n-gram fallback
        logging.debug(
            " %s Pattern extraction validation failed, trying n-gram fallback...", EMOJI_CROSS)
    else:
        logging.debug(
            " %s No building pattern match found in query, trying n-gram fallback...", EMOJI_CROSS)

    # Phase 2: N-gram fallback for queries that don't match patterns
    # This is especially useful for lowercase queries like "does old park hill have an fra?"
    return extract_building_from_query_ngram_fallback(query, known_buildings)


def validate_building_name_fuzzy(
    extracted_name: str,
    known_buildings: Optional[list[str]] = None
) -> Optional[str]:
    """
    Validate extracted name against known buildings using fuzzy matching.
    IMPROVED: Checks all metadata field variations with 80% threshold and
    rejects invalid/maintenance-like building names consistently.

    Args:
        extracted_name: Extracted building name candidate
        known_buildings: List of known building names

    Returns:
        Canonical building name if matched, None otherwise
    """
    if not extracted_name:
        return None

    # Use cache if no explicit list provided
    if known_buildings is None:
        if BuildingCacheManager.is_populated():
            known_buildings = get_building_names_from_cache()
        else:
            return None

    if not known_buildings:
        return None

    # Cache the lowercase version for multiple comparisons
    extracted_lower = normalise_building_name(extracted_name)

    # 🔒 Hard rejection of maintenance-context / invalid tokens
    # (e.g. "maintenance", "request", "job", etc.)
    if extracted_lower in INVALID_BUILDING_NAMES:
        logging.info(
            "%s Rejecting '%s' as building (maintenance/invalid keyword)",
            EMOJI_CROSS,
            extracted_lower,
        )
        return None

    # extracted_norm = normalise_building_name(extracted_name)

    # FILTER: Reject if the extracted name is a common query term
    if extracted_lower in QUESTION_WORDS:
        logging.debug(
            "%s Rejected '%s' - matches common query term",
            EMOJI_CROSS,
            extracted_name,
        )
        return None

    # 🔒 Filter known buildings to valid ones only (avoid canonical "Maintenance")
    valid_buildings: list[str] = [
        b for b in known_buildings
        if is_valid_building_name(b)
    ]
    if not valid_buildings:
        return None
    known_buildings = valid_buildings

    # Strategy 1: Exact match in aliases cache
    if BuildingCacheManager.is_populated():
        canonical = _BUILDING_ALIASES_CACHE.get(extracted_lower)
        if canonical and is_valid_building_name(canonical):
            logging.info(
                "%s Alias exact match: '%s' -> '%s'",
                EMOJI_TICK,
                extracted_name,
                canonical,
            )
            return canonical

        # Check canonical names cache
        canonical = _BUILDING_NAMES_CACHE.get(extracted_lower)
        if canonical and is_valid_building_name(canonical):
            logging.info(
                "%s Canonical exact match: '%s'",
                EMOJI_TICK,
                canonical,
            )
            return canonical

    # Strategy 2: Exact match (case-insensitive) in known buildings
    for building in known_buildings:
        if building.lower().strip() == extracted_lower:
            # is_valid_building_name already enforced in known_buildings
            logging.info("%s Exact match: '%s'", EMOJI_TICK, building)
            return building

    # Strategy 3: Substring match (extracted name in building name)
    for building in known_buildings:
        if extracted_lower in building.lower():
            logging.info(
                "%s Substring match: '%s' in '%s'",
                EMOJI_TICK,
                extracted_name,
                building,
            )
            return building

    # Strategy 4: Reverse substring match (building name in extracted name)
    for building in known_buildings:
        if building.lower() in extracted_lower:
            logging.info(
                "%s Reverse substring match: '%s' contains '%s'",
                EMOJI_TICK,
                extracted_name,
                building,
            )
            return building

    # Strategy 5: Fuzzy match against all metadata field variations
    if BuildingCacheManager.is_populated():
        best_match = None
        best_score = 0.0
        best_canonical = None

        for canonical in known_buildings:
            match_result = fuzzy_match_against_metadata_fields(
                extracted_name,
                canonical,
                FUZZY_MATCH_THRESHOLD,
            )

            if match_result:
                matched_field, score = match_result
                if score > best_score and is_valid_building_name(canonical):
                    best_score = score
                    best_match = matched_field
                    best_canonical = canonical

        if best_canonical:
            logging.info(
                "%s Fuzzy match (%.1f%%): '%s' -> '%s' (via field '%s')",
                EMOJI_TICK,
                best_score * 100,
                extracted_name,
                best_canonical,
                best_match,
            )
            return best_canonical

    # Strategy 6: Standard fuzzy match using difflib (e.g. ≥ 80% similarity)
    matches = get_close_matches(
        extracted_name,
        known_buildings,
        n=1,
        cutoff=FUZZY_MATCH_THRESHOLD,
    )
    if matches:
        candidate = matches[0]
        if is_valid_building_name(candidate):
            logging.info(
                "%s Difflib fuzzy match (≥%.0f%%): '%s' -> '%s'",
                EMOJI_TICK,
                FUZZY_MATCH_THRESHOLD * 100,
                extracted_name,
                candidate,
            )
            return candidate

    logging.debug("%s No match found for '%s'", EMOJI_CROSS, extracted_name)
    return None


# ============================================================================
# RESULT PROCESSING FUNCTIONS
# ============================================================================


def group_results_by_building(
    results: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """
    Group search results by normalised building name.
    IMPROVED: Uses enhanced building name extraction from multiple fields.
    """
    grouped = {}

    for result in results:
        # Extract building name from multiple metadata fields
        metadata = result.get('metadata', {})
        building_name = None

        # Check metadata fields in priority order
        for field in ['canonical_building_name', 'building_name', 'Property names', 'UsrFRACondensedPropertyName']:
            value = metadata.get(field) or result.get(field)
            if value:
                if isinstance(value, list):
                    # Take first non-empty item from list
                    for item in value:
                        if item and str(item).strip():
                            building_name = str(item).strip()
                            break
                elif isinstance(value, str) and value.strip():
                    building_name = value.strip()

                if building_name:
                    break

        # Fallback if no building name found
        if not building_name or building_name == 'Unknown':
            building_name = result.get('building_name', 'Unknown')

        normalised_name = normalise_building_name(building_name)

        if not normalised_name:
            normalised_name = building_name

        if normalised_name not in grouped:
            grouped[normalised_name] = []

        if '_normalised_building' not in result:
            result['_normalised_building'] = normalised_name

        # Also store the original building name for reference
        if '_original_building' not in result:
            result['_original_building'] = building_name

        grouped[normalised_name].append(result)

    return grouped


def result_matches_building(result: dict[str, Any], target_building: str) -> bool:
    """
    Check if a search result belongs to the target building using fuzzy + alias logic.

    Args:
        result: Search result dictionary (with metadata)
        target_building: Building name to check against

    Returns:
        True if the result matches the building, otherwise False
    """
    if not target_building:
        return True

    # Normalised canonical form of target
    target_norm = normalise_building_name(target_building)

    metadata = result.get("metadata", {})
    candidate_values = []

    # Collect all potential building-related values from metadata
    for field in BUILDING_METADATA_FIELDS:
        value = metadata.get(field) or result.get(field)
        if not value:
            continue
        if isinstance(value, list):
            candidate_values.extend(value)
        else:
            candidate_values.append(value)

    # Compare each value to the target building using fuzzy validation
    for candidate in candidate_values:
        validated = validate_building_name_fuzzy(str(candidate))
        if not validated:
            continue
        candidate_norm = normalise_building_name(validated)
        if candidate_norm == target_norm:
            return True

    return False


def filter_results_by_building(results: list[dict[str, Any]],
                               target_building: str) -> list[dict[str, Any]]:
    """
    Filter search results to only include those matching the target building.
    """
    if not target_building:
        return results

    filtered = [r for r in results if result_matches_building(
        r, target_building)]

    logging.info(
        "🏢 Filtered %d/%d results for building '%s'",
        len(filtered),
        len(results),
        target_building
    )

    return filtered


def prioritise_building_results(
    results: list[dict[str, Any]],
    target_building: str
) -> list[dict[str, Any]]:
    """
    Reorder results to prioritise a specific building.
    """
    if not target_building or not results:
        return results

    target_normalised = normalise_building_name(target_building).lower()
    target_lower = target_building.lower()

    priority_results = []
    other_results = []

    for result in results:
        building_name = result.get('building_name', '')

        if '_normalised_building' in result:
            normalised = result['_normalised_building'].lower()
        else:
            normalised = normalise_building_name(building_name).lower()

        building_lower = building_name.lower()

        is_match = (
            target_normalised in normalised or
            normalised in target_normalised or
            target_lower in building_lower or
            building_lower in target_lower
        )

        if is_match:
            priority_results.append(result)
        else:
            other_results.append(result)

    logging.info(
        "Prioritised %d results for '%s', %d other results",
        len(priority_results),
        target_building,
        len(other_results)
    )

    return priority_results + other_results


def get_building_context_summary(
    building_results: dict[str, list[dict[str, Any]]]
) -> str:
    """
    Create a summary of buildings found in search results.
    """
    if not building_results:
        return ""

    summary_parts = []
    for building, results in building_results.items():
        doc_types = {}
        for r in results:
            doc_type = r.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        type_str = ', '.join(
            f"{count} {dtype}" for dtype, count in sorted(doc_types.items())
        )
        summary_parts.append(
            f"- {building}: {len(results)} results ({type_str})"
        )

    return "Buildings found:\n" + "\n".join(summary_parts)


# ============================================================================
# CACHE UTILITIES
# ============================================================================


def clear_caches():
    """Clear all LRU caches."""
    normalise_building_name.cache_clear()
    clear_building_cache()
    logging.info("Cleared building_utils caches")


def get_cache_info() -> dict[str, Any]:
    """Get information about cache usage for monitoring."""
    return {
        'normalise_building_name': normalise_building_name.cache_info()._asdict(),
        'building_cache': get_cache_status()
    }


# ============================================================================
# DEBUGGING/TESTING UTILITIES
# ============================================================================


def test_building_extraction(test_queries: list[str]) -> dict[str, Optional[str]]:
    """
    Test building extraction on multiple queries.

    Args:
        test_queries: List of query strings to test

    Returns:
        Dictionary mapping queries to extracted buildings
    """
    results = {}
    for query in test_queries:
        building = extract_building_from_query(query)
        results[query] = building
        logging.info("Test: '%s' -> %s", query, building or "None")

    return results


def get_building_metadata_summary(canonical_name: str) -> dict[str, Any]:
    """
    Get summary of all metadata field variations for a building.

    Args:
        canonical_name: Canonical building name

    Returns:
        Dictionary with metadata field information
    """
    if canonical_name not in _METADATA_FIELDS_CACHE:
        return {
            'canonical_name': canonical_name,
            'found': False,
            'metadata_fields': []
        }

    return {
        'canonical_name': canonical_name,
        'found': True,
        'metadata_fields': sorted(list(_METADATA_FIELDS_CACHE[canonical_name])),
        'field_count': len(_METADATA_FIELDS_CACHE[canonical_name])
    }
