from typing import Optional, TypedDict
import re
from date_utils import parse_date_to_iso


class FRAMetadata(TypedDict, total=False):
    fra_assessment_date: Optional[str]
    fra_assessor: Optional[str]


STRICT_ASSESSMENT_DATE_PATTERNS = [
    r"Date of fire risk assessment:\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
    r"Date of fire risk assessment:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    r"Assessment date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
]

ASSESSOR_PATTERNS = [
    r"Fire risk assessor:\s*(.+)",
    r"Assessor:\s*(.+)",
    r"Assessment carried out by:\s*(.+)",
]


def _sanitise_assessor(value: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    if not cleaned:
        return None
    # Remove trailing punctuation/noise
    cleaned = re.sub(r"[\s\.,;:]+$", "", cleaned)
    # Drop placeholder values
    placeholders = {"n/a", "na", "unknown", "tbd", "none", "-"}
    if cleaned.strip().lower() in placeholders:
        return None
    # Truncate to a reasonable length to avoid garbage blobs
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()
    return cleaned


def extract_assessment_date(text: str) -> Optional[str]:
    """Extract assessment date from document text using strict patterns."""
    for pattern in STRICT_ASSESSMENT_DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_date_to_iso(match.group(1).strip())
    return None


def extract_fra_metadata(text: str) -> FRAMetadata:
    metadata: FRAMetadata = {}

    metadata["fra_assessment_date"] = extract_assessment_date(text)

    for pattern in ASSESSOR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["fra_assessor"] = _sanitise_assessor(match.group(1))
            break

    return metadata
