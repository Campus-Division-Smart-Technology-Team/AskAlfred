#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small, isolated text-based building extraction used as a fallback resolver.
"""
from __future__ import annotations
import re
from difflib import get_close_matches
# Third party import
from config import (
    BUILDING_TEXT_CUTOFF,
    BUILDING_TEXT_FIRST_LINES,
    BUILDING_TEXT_NGRAM_MAX,
    BUILDING_TEXT_NGRAM_MIN,
)
# First party import
from .normaliser import normalise_building_name


def extract_building_from_text(
    text_sample: str,
    name_to_canonical: dict[str, str],
    alias_to_canonical: dict[str, str],
    known_buildings: list[str],
) -> str:
    """Extract building name from text sample using simple matching."""
    if not text_sample:
        return "Unknown"

    # Lowercase once; we still normalise keys for robust substring checks
    first_lines = " ".join((text_sample or "").split(
        "\n")[:BUILDING_TEXT_FIRST_LINES]).lower()

    # Build normalised maps
    alias_norm = {normalise_building_name(
        k): v for k, v in (alias_to_canonical or {}).items()}
    name_norm = {normalise_building_name(
        k): v for k, v in (name_to_canonical or {}).items()}

    # 1) Exact alias hits (normalised contains check)
    for alias_key, canonical in alias_norm.items():
        if alias_key and alias_key in first_lines:
            return canonical

    # 2) Canonical variants
    for name_key, canonical in name_norm.items():
        if name_key and name_key in first_lines:
            return canonical

    # 3) Light fuzzy on sliding n-grams from early lines
    words = re.sub(r"[^a-z0-9\s]", " ", first_lines).split()
    for n in range(BUILDING_TEXT_NGRAM_MAX, BUILDING_TEXT_NGRAM_MIN - 1, -1):
        for i in range(0, max(0, len(words) - n + 1)):
            candidate = " ".join(words[i:i+n])
            matches = get_close_matches(
                candidate, known_buildings, n=1, cutoff=BUILDING_TEXT_CUTOFF)
            if matches:
                return matches[0]

    return "Unknown"


if __name__ == "__main__":
    sample = """Fire Risk Assessment for the Churchill Hall A - B site\nMeans of Escape …"""
    print(extract_building_from_text(sample, {}, {}, ["Churchill Hall A - B"]))
