"""
Building utilities package exports.
"""

from .alias_override import get_alias_override
from .normaliser import normalise_building_name
from .resolver import BuildingResolution, BuildingResolver
from .text_fallback import extract_building_from_text
from .utils import (
    BuildingCacheManager,
    clear_building_cache,
    extract_building_from_query,
    get_building_names_from_cache,
    get_cache_status,
    result_matches_building,
)
from .validation import (
    INVALID_BUILDING_NAMES,
    is_valid_building_name,
    sanitise_building_candidate,
)
from .cache import (
    BUILDING_ALIASES_CACHE,
    BUILDING_NAMES_CACHE,
    INDEXES_WITH_BUILDINGS,
    METADATA_FIELDS_CACHE,
    clear_all_building_cache,
)

__all__ = [
    "get_alias_override",
    "normalise_building_name",
    "BuildingResolution",
    "BuildingResolver",
    "extract_building_from_text",
    "BuildingCacheManager",
    "clear_building_cache",
    "extract_building_from_query",
    "get_building_names_from_cache",
    "get_cache_status",
    "result_matches_building",
    "INVALID_BUILDING_NAMES",
    "is_valid_building_name",
    "sanitise_building_candidate",
    "BUILDING_ALIASES_CACHE",
    "BUILDING_NAMES_CACHE",
    "INDEXES_WITH_BUILDINGS",
    "METADATA_FIELDS_CACHE",
    "clear_all_building_cache",
]
