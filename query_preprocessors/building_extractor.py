# building_extractor.py

from query_preprocessors.base_preprocessor import CachingPreprocessor
from building_utils import (
    extract_building_from_query,
    get_building_names_from_cache,
    BuildingCacheManager
)
from emojis import EMOJI_TICK


class BuildingExtractor(CachingPreprocessor):
    """
    Uses authoritative building_utils.extract_building_from_query()
    to extract:
        context.buildings = [list of matches]
        context.building  = first match (legacy)
        context.building_filter = same
    Mark:
        context.cache["building_detected"] = True
    """

    order = 20
    cache_key = "building_extracted"

    def process(self, context):
        query = context.query

        if not BuildingCacheManager.is_populated():
            self.logger.warning(
                "Building cache not populated — skipping extraction")
            return

        known_buildings = get_building_names_from_cache()
        if not known_buildings:
            return

        # AUTHORITATIVE function call
        detected = extract_building_from_query(query, known_buildings)

        if not detected:
            return

        # Multi-building support:
        context.buildings = [detected]  # Expandable later
        context.building = detected
        context.building_filter = detected

        context.add_to_cache("building_detected", True)
        self.mark_done(context)

        self.logger.info("%s Detected building: %s", EMOJI_TICK, detected)
