#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for building-name normalisation.
Import this module everywhere you need to compare / map building names.

Guidelines
---------
* Always normalise before using a name as a dict key or for matching.
* Normalisation is **lossy** and intended only for matching/lookup, not display.
* Keep this file tiny, deterministic, and well-tested.
"""
from __future__ import annotations

from functools import lru_cache
import re
from collections.abc import Iterable


__all__ = ["normalise_building_name", "normalise_key", "normalise_keys",]


# Common suffixes we strip off for lookups (display names remain untouched).
_SUFFIXES = (" building", " house", " hall", " centre", " center",
             " complex", " tower", " block", " wing", " facility",)


# Collapse multiple whitespace
_ws = re.compile(r"\s+")


@lru_cache(maxsize=2048)
def normalise_building_name(name: str | None) -> str:
    """Return a normalised form of *name* for consistent lookup.


    This function is intentionally conservative. It lowercases, trims, removes
    a small set of generic suffixes, collapses whitespace, and preserves
    digits and symbols that materially distinguish a site (e.g. "@", "/").


    Parameters
    ----------
    name : str | None
    Raw building name or alias. `None` yields an empty string.


    Returns
    -------
    str
    Normalised key suitable for matching and dictionary keys.
    """
    if not name:
        return ""

    s = str(name).strip().lower()

    # Strip known generic suffixes (if present) **once**
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].rstrip()
            break

    # Collapse whitespace
    s = _ws.sub(" ", s)

    return s


def normalise_key(name: str | None) -> str:
    """Alias for :func:`normalise_building_name` for readability in maps."""
    return normalise_building_name(name)


def normalise_keys(names: Iterable[str]) -> list[str]:
    """Normalise a sequence of names (helper for tests and map-building)."""
    return [normalise_key(n) for n in names]


if __name__ == "__main__":  # quick smoke-tests
    cases = {
        "Physics Building": "physics",
        "Retort House": "retort",
        "Churchill Hall": "churchill",
        " Indoor Sports Centre ": "indoor sports centre",
        "Accommodation@33": "accommodation@33",
        "71 St Michaels Hill": "71 st michaels hill",
        "Maintenance / Garden Store": "maintenance / garden store",
    }
    for raw, want in cases.items():
        got = normalise_building_name(raw)
        assert got == want, (raw, got, want)
    print("✅ building_normaliser smoke-tests passed")
