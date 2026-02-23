# building_extractor.py

from query_preprocessors.base_preprocessor import CachingPreprocessor
from building.utils import (
    extract_building_from_query,
    get_building_names_from_cache,
    BuildingCacheManager
)
from building.validation import is_valid_building_name
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
            BuildingCacheManager.ensure_initialised()

        if not BuildingCacheManager.is_populated():
            self.logger.error("Building cache failed to initialise")
            context.add_to_cache("building_detected", False)
            return

        known_buildings = get_building_names_from_cache()
        if not known_buildings:
            context.add_to_cache("building_detected", False)
            return

        # AUTHORITATIVE function call
        detected = extract_building_from_query(query, known_buildings)

        # explicitly record “no valid building” so QueryManager can
        # still inherit from previous turn.
        if not detected:
            context.add_to_cache("building_detected", False)
            self.logger.debug("No building detected in query")
            return

        if not is_valid_building_name(detected):
            self.logger.info(
                "⚠️ Ignoring invalid building candidate from extractor: %s",
                detected,
            )
            context.add_to_cache("building_detected", False)
            context.add_to_cache("building_invalid_candidate", detected)
            # Don't set context.building at all — keep context clean so
            # downstream can safely inherit previous building.
            return

        # Multi-building support:
        context.buildings = [detected]  # Expandable later
        context.building = detected
        context.building_filter = detected

        context.add_to_cache("building_detected", True)
        self.mark_done(context)

        self.logger.info("%s Detected building: %s", EMOJI_TICK, detected)
