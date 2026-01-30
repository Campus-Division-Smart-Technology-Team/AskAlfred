from __future__ import annotations
from typing import Optional
import re

# INVALID_BUILDING_NAMES = frozenset({"maintenance","request", "requests","job", "jobs","ticket", "tickets",})

INVALID_BUILDING_NAMES: frozenset[str] = frozenset({
    "maintenance",
    "request",
    "requests",
    "job",
    "jobs",
    "building",
    "property",
    "data",
    "planon",
    "bms",
    "fra",
    "fire",
    "risk",
    "assessment",
    "system",
    "management",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "does",
    "have",
    "has",
    "the",
    "this",
    "that",
}
)


def is_valid_building_name(name: str | None) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    if not s:
        return False
    # only block stopwords if it’s a single token
    if len(s.split()) == 1 and s in INVALID_BUILDING_NAMES:
        return False
    return True


_PUNCT_RE = re.compile(r"[^\w\s\-']+", re.UNICODE)


def normalise_candidate_building(name: str | None) -> str:
    """
    Normalise a candidate building string for validation checks.
    - lowercases
    - strips whitespace
    - removes most punctuation (keeps hyphen/apostrophe)
    - collapses multiple spaces
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def is_valid_building_candidate(name: str | None) -> bool:
    """
    Conservative validator for extracted building 'candidates'.
    Returns True only if the string looks like a plausible building name.
    """
    s = normalise_candidate_building(name)
    if not s:
        return False

    tokens = s.split()

    # If it's a single token, block common stopwords / junk
    if len(tokens) == 1 and s in INVALID_BUILDING_NAMES:
        return False

    # Block obviously meaningless very short fragments
    if len(s) < 3:
        return False

    return True


def sanitise_building_candidate(name: str | None) -> Optional[str]:
    """
    Returns the original name if it's valid, otherwise None.
    Keeps original casing for display purposes.
    """
    return name if is_valid_building_candidate(name) else None
