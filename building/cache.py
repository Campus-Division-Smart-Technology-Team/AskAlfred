# buildings_cache.py
"""
Shared building cache utilities to avoid circular imports.
"""

# Caches
_CACHE_STATE = {
    "populated": False,
}
BUILDING_NAMES_CACHE: dict[str, str] = {}  # normalised → canonical
BUILDING_ALIASES_CACHE: dict[str, str] = {}  # alias → canonical
METADATA_FIELDS_CACHE: dict[str, set[str]] = {}
# Reverse index over every known metadata variation/alias (lowercased) so hot
# paths can resolve a candidate string to its canonical building in O(1)
# instead of fuzzy-matching against every building.
BUILDING_VARIATIONS_REVERSE: dict[str, str] = {}  # variation (lower) → canonical
INDEXES_WITH_BUILDINGS: list[str] = []


def register_building_variation(variation: str, canonical: str) -> None:
    """Record a variation → canonical mapping for O(1) reverse lookups."""
    key = (variation or "").strip().lower()
    if key:
        BUILDING_VARIATIONS_REVERSE.setdefault(key, canonical)


def clear_all_building_cache():
    BUILDING_NAMES_CACHE.clear()
    BUILDING_ALIASES_CACHE.clear()
    METADATA_FIELDS_CACHE.clear()
    BUILDING_VARIATIONS_REVERSE.clear()
    INDEXES_WITH_BUILDINGS.clear()
    _CACHE_STATE["populated"] = False
