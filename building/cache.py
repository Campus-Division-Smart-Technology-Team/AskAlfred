# buildings_cache.py
"""
Shared building cache utilities to avoid circular imports.
"""

# Caches
_CACHE_STATE = {
    "populated": False,
}
BUILDING_NAMES_CACHE: dict[str, str] = {}   # normalised → canonical
BUILDING_ALIASES_CACHE: dict[str, str] = {}  # alias → canonical
METADATA_FIELDS_CACHE: dict[str, set[str]] = {}
INDEXES_WITH_BUILDINGS: list[str] = []


def clear_all_building_cache():
    BUILDING_NAMES_CACHE.clear()
    BUILDING_ALIASES_CACHE.clear()
    METADATA_FIELDS_CACHE.clear()
    INDEXES_WITH_BUILDINGS.clear()
    _CACHE_STATE["populated"] = False
