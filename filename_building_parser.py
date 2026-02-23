#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename-based extraction with centralised normalisation.


Key changes from your current module:
- Uses `building_normaliser.normalise_building_name` before **any** lookup.
- Looks up against *normalised* views of alias/name maps.
- Still respects `building_aliases_override.get_alias_override`.
"""
from __future__ import annotations

import logging
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Optional, Tuple, List

from building_normaliser import normalise_building_name

# --------------------------------------------------------------------------------------
# Matching thresholds
# --------------------------------------------------------------------------------------

FUZZY_STRONG = 0.75   # strong match from filename
FUZZY_WEAK = 0.50   # weaker filename fallback if nothing stronger
TEXT_CUTOFF = 0.80   # fuzzy cutoff for text n-grams

# --------------------------------------------------------------------------------------
# Core filename parsing
# --------------------------------------------------------------------------------------


def extract_building_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None

    name = filename.rsplit('.', 1)[0].split('/')[-1]

    # Pattern 1: [Optional PREFIX-]FRA-<Building>-YYYY-MM(-DD)?
    fra_match = re.match(
        r'(?:[A-Z]+-)?FRA-(?P<building>.+?)-.*?(\d{4}-\d{2}(?:-\d{2})?)',
        name, flags=re.IGNORECASE)

    if fra_match:
        building = fra_match.group('building')
        building = building.replace('-', ' ')
        building = re.sub(r'([a-z])([A-Z])', r'\1 \2', building)
        building = re.sub(r'(\d+)([A-Z])', r'\1 \2', building)
        return re.sub(r'\s+', ' ', building).strip()

    # Pattern 1B: FM-OAS FRA-style filenames
    # Example: FM-OAS-RichmondBuilding-2024-02
    oas_match = re.match(
        r'(?:FM|RFM|SRL)-OAS-(?P<building>.+?)-(\d{4}-\d{2}(?:-\d{2})?)',
        name,
        flags=re.IGNORECASE
    )
    if oas_match:
        building = oas_match.group('building')
        building = building.replace('-', ' ')
        building = re.sub(r'([a-z])([A-Z])', r'\1 \2', building)
        building = re.sub(r'(\d+)([A-Z])', r'\1 \2', building)
        return re.sub(r'\s+', ' ', building).strip()

    # Pattern 2: UoB-<Building>-(BMS|HVAC|FRA|O&M|OM|Controls|Manual|Project)
    uob_match = re.match(
        r'UoB-(.+?)-(?:BMS|HVAC|FRA|O&M|OM|Controls|Manual|Project)', name, re.IGNORECASE,)
    if uob_match:
        building = uob_match.group(1)
        building = re.sub(r'-Project$', '', building, flags=re.IGNORECASE)
        building = building.replace('-', ' ')
        building = re.sub(r'([a-z])of([A-Z])', r'\1 of \2', building)
        building = re.sub(r'([a-z])([A-Z])', r'\1 \2', building)
        return re.sub(r'\s+', ' ', building).strip()

    # Pattern 3: Catch FM/OAS style filenames: PREFIX-PREFIX2-BUILDING-DATE
    p3 = re.search(
        r'^[A-Z]{2,5}-[A-Z]{2,5}-(?P<bldg>.+?)-\d{4}-\d{2}(?:-\d{2})?$',
        name,
        flags=re.IGNORECASE
    )
    if p3:
        building = p3.group('bldg')
        building = building.replace('-', ' ')
        building = re.sub(r'([a-z])([A-Z])', r'\1 \2', building)
        building = re.sub(r'(\d+)([A-Z])', r'\1 \2', building)
        return re.sub(r'\s+', ' ', building).strip()

    return None

# --------------------------------------------------------------------------------------
# Matching utilities
# --------------------------------------------------------------------------------------


def _fuzzy_to_known(extracted_name: str, known_buildings: List[str], threshold: float = 0.75) -> Tuple[Optional[str], float]:
    if not extracted_name or not known_buildings:
        return None, 0.0
    extracted_lower = extracted_name.lower()
    known_lower = [b.lower() for b in known_buildings]

    if extracted_lower in known_lower:
        idx = known_lower.index(extracted_lower)
        return known_buildings[idx], 1.0

    match = get_close_matches(
        extracted_lower, known_lower, n=1, cutoff=threshold)
    if match:
        idx = known_lower.index(match[0])
        score = SequenceMatcher(None, extracted_lower, match[0]).ratio()
        return known_buildings[idx], score

    return None, 0.0

# --------------------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------------------


def get_building_with_confidence(
        filename: str,
        text_sample: str,
        known_buildings: list[str],
        name_to_canonical: dict[str, str],
        alias_to_canonical: dict[str, str],) -> tuple[Optional[str], float, str]:
    """
    Resolve a canonical building for a document with a confidence score.


    Strategy order:
    1) Filename → alias overrides → exact alias/name maps (all 1.0)
    2) Filename → fuzzy to known buildings (>= 0.75)
    3) Text-based fallback (0.60)
    4) Unknown
    """
    extracted_raw = extract_building_from_filename(filename)

    if extracted_raw:
        extracted_norm = normalise_building_name(extracted_raw)

        # 1a) Project-level hard overrides
        try:
            from building_aliases_override import get_alias_override  # pylint: disable=import-outside-toplevel
            override = get_alias_override(extracted_norm)
            if override:
                logging.info("alias override: %s → %s",
                             extracted_raw, override)
                return override, 1.0, 'filename'
        except ImportError:
            pass

        # Build normalised views of your maps **once**
        alias_norm = {normalise_building_name(
            k): v for k, v in (alias_to_canonical or {}).items()}
        name_norm = {normalise_building_name(
            k): v for k, v in (name_to_canonical or {}).items()}

        # 1b) Exact alias/name lookups on normalised keys
        if extracted_norm in alias_norm:
            return alias_norm[extracted_norm], 1.0, 'filename'
        if extracted_norm in name_norm:
            return name_norm[extracted_norm], 1.0, 'filename'

        # 2) Fuzzy to known canonicals
        matched, score = _fuzzy_to_known(
            extracted_raw, known_buildings, threshold=0.75)
        if matched and score >= FUZZY_STRONG:
            return matched, score, 'filename'
        if matched and score >= FUZZY_WEAK:
            return matched, score, 'filename'

    # 3) Text fallback – delegate to local_batch_ingest helper (kept for compatibility)
    try:
        from local_batch_ingest_text_fallback import extract_building_from_text
        guess = extract_building_from_text(
            text_sample, name_to_canonical, alias_to_canonical, known_buildings)
        if guess and guess != "Unknown":
            return guess, 0.60, 'text'
    except Exception as e:
        logging.debug("text fallback failed: %s", e)

    # 4) Unknown
    return "Unknown", 0.0, 'unknown'


# Convenience for downstream use
def should_flag_for_review(conf, src):
    return (src == 'unknown' or (src == 'text' and conf < 0.70) or (src == 'filename' and conf < 0.75))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    examples = [
        "RFM-FRA-ChurchillA-B-2023-10.pdf",
        "FM-FRA-Garden Store-FRA-2025-10-08.pdf",
        "UoB-SchoolofDentistry-BMS-P7391-OM-As-Installed.pdf",
    ]
    for f in examples:
        print(f, "→", extract_building_from_filename(f))
