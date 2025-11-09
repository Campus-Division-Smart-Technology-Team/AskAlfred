# buildings_cache.py
"""
Shared building cache utilities to avoid circular imports.
"""

from typing import Dict, Set, List

# Caches
BUILDING_NAMES_CACHE: Dict[str, str] = {}   # normalised → canonical
BUILDING_ALIASES_CACHE: Dict[str, str] = {}  # alias → canonical
METADATA_FIELDS_CACHE: Dict[str, Set[str]] = {}
CACHE_POPULATED: bool = False
INDEXES_WITH_BUILDINGS: List[str] = []


def clear_all_building_cache():
    BUILDING_NAMES_CACHE.clear()
    BUILDING_ALIASES_CACHE.clear()
    METADATA_FIELDS_CACHE.clear()
    INDEXES_WITH_BUILDINGS.clear()
    global CACHE_POPULATED
    CACHE_POPULATED = False
