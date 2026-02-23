#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Building Aliases Override Configuration

This file provides exact name mappings for buildings where fuzzy matching fails
due to low similarity scores. Use this as a workaround when you cannot modify
the Property CSV directly.

INSTRUCTIONS:
1. Copy this file to your project directory (same folder as local_batch_ingest.py)
2. Modify filename_building_parser.py to check these overrides first
3. Verify all mappings are correct before deploying
4. Clear cache and re-ingest documents

IMPORTANT: This is a WORKAROUND. The proper solution is to add these aliases
to your Property CSV in the "Property alternative names" column.
"""

from typing import Optional, Dict

# ============================================================================
# ALIAS MAPPINGS
# ============================================================================

# Exact mappings: extracted name (lowercase) -> canonical property name
# These are checked BEFORE fuzzy matching, giving 100% confidence
ALIAS_OVERRIDES: Dict[str, str] = {
    # =========================================================================
    # CONFIRMED MAPPINGS (All verified and ready to use!)
    # =========================================================================

    # Garden Store → Maintenance / Garden Store (Property Code: 202)
    # Similarity score: 63.2% (too low for fuzzy match)
    'garden store': 'Maintenance / Garden Store',

    # Maintenance Office & Workshop → Maintenance / Garden Store (Property Code: 202)
    # Same building as Garden Store
    'maintenance office & workshop': 'Maintenance / Garden Store',
    'maintenance office and workshop': 'Maintenance / Garden Store',

    # Churchill A B → Churchill Hall A - B (Property Code: 539)
    # Similarity score: 78.8% (below 80% threshold)
    'churchill a b': 'Churchill Hall A - B',
    'churchilla-b': 'Churchill Hall A - B',  # Alternative format
    'churchill a-b': 'Churchill Hall A - B',  # Another variation

    # Indoor Sports Hall → Indoor Sports Centre (Property Code: 135)
    # Similarity score: 73.7% (below 80% threshold)
    'indoor sports hall': 'Indoor Sports Centre',

    # Accommodation@33 → Accommodation at Thirty-Three (Property Code: 59)
    # Similarity score: 57.8% (too low for fuzzy match)
    # ✅ CONFIRMED by user
    'accommodation@33': 'Accommodation at Thirty-Three',
    'accommodation 33': 'Accommodation at Thirty-Three',
    'accommodation at 33': 'Accommodation at Thirty-Three',

    # School of Dentistry → Bristol Dental Hospital (Property Code: 122)
    # Similarity score: 47.6% (way too low for fuzzy match)
    # ✅ CONFIRMED by user
    'school of dentistry': 'Bristol Dental Hospital',
    'dentistry school': 'Bristol Dental Hospital',

    # Whiteladies → 1-5 Whiteladies Road (Property Code: 744)
    # ✅ CONFIRMED by user (not 7-9 Whiteladies Road)
    'whiteladies': '1-5 Whiteladies Road',

    # Retort House → 65 Avon Street (Property Code: 951)
    # NOTE: This alias should already exist in CSV, but including as backup
    'retort house': '65 Avon Street',

    # =========================================================================
    # ADD YOUR OWN MAPPINGS HERE
    # =========================================================================
    # Format: 'extracted name (lowercase)': 'Canonical Property Name',

    # Example:
    # 'engineering building': 'Engineering Faculty Building',
    # 'main library': 'Arts and Social Sciences Library',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_alias_override(extracted_name: str) -> Optional[str]:
    """
    Get canonical building name from alias overrides.

    This function is called BEFORE fuzzy matching to provide exact matches
    for buildings where fuzzy matching fails due to low similarity scores.

    Args:
        extracted_name: Building name extracted from filename (case-insensitive)

    Returns:
        Canonical building name if override exists, None otherwise

    Examples:
        >>> get_alias_override("Garden Store")
        'Maintenance / Garden Store'

        >>> get_alias_override("Churchill A B")
        'Churchill Hall A - B'

        >>> get_alias_override("Unknown Building")
        None
    """
    if not extracted_name:
        return None

    # Normalise to lowercase for lookup
    normalized = extracted_name.lower().strip()

    return ALIAS_OVERRIDES.get(normalized)


def list_all_overrides() -> Dict[str, str]:
    """
    Get all defined alias overrides.

    Returns:
        Dictionary of all alias mappings
    """
    return ALIAS_OVERRIDES.copy()


def add_alias_override(extracted_name: str, canonical_name: str) -> None:
    """
    Add a new alias override (useful for dynamic configuration).

    Args:
        extracted_name: Building name as extracted from filename
        canonical_name: Canonical property name from CSV

    Example:
        >>> add_alias_override("New Building", "New Academic Building")
    """
    ALIAS_OVERRIDES[extracted_name.lower().strip()] = canonical_name


def verify_override_exists(canonical_name: str, property_csv_path: str) -> bool:
    """
    Verify that a canonical building name exists in the Property CSV.

    This helps catch typos in override mappings.

    Args:
        canonical_name: The canonical property name to verify
        property_csv_path: Path to Dim-Property.csv file

    Returns:
        True if the building exists in CSV, False otherwise
    """
    try:
        import pandas as pd
        df = pd.read_csv(property_csv_path)
        return canonical_name in df['Property name'].values
    except Exception as e:
        print(f"Warning: Could not verify override: {e}")
        return False


def generate_override_report() -> str:
    """
    Generate a formatted report of all alias overrides.

    Returns:
        Formatted string report
    """
    report_lines = [
        "="*80,
        "BUILDING ALIAS OVERRIDES REPORT",
        "="*80,
        "",
        f"Total overrides defined: {len(ALIAS_OVERRIDES)}",
        "",
        "Mappings:",
        "-"*80,
    ]

    for extracted, canonical in sorted(ALIAS_OVERRIDES.items()):
        report_lines.append(f"  '{extracted}' → '{canonical}'")

    report_lines.extend([
        "",
        "="*80,
        "",
        "These mappings bypass fuzzy matching and provide 100% confidence matches.",
        "They are checked BEFORE any other lookup method.",
        "",
        "To modify these mappings, edit building_aliases_override.py",
        "="*80,
    ])

    return "\n".join(report_lines)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_overrides(property_csv_path: str) -> Dict[str, bool]:
    """
    Validate all override mappings against Property CSV.

    Args:
        property_csv_path: Path to Dim-Property.csv

    Returns:
        Dictionary mapping canonical names to validation status
    """
    import pandas as pd

    try:
        df = pd.read_csv(property_csv_path)
        property_names = set(df['Property name'].values)

        validation_results = {}
        for extracted, canonical in ALIAS_OVERRIDES.items():
            validation_results[canonical] = canonical in property_names

        return validation_results
    except Exception as e:
        print(f"Error validating overrides: {e}")
        return {}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Print the override report
    print(generate_override_report())

    # Test some lookups
    print("\n" + "="*80)
    print("TESTING OVERRIDE LOOKUPS")
    print("="*80 + "\n")

    test_cases = [
        "Garden Store",
        "Churchill A B",
        "Indoor Sports Hall",
        "Retort House",
        "Unknown Building",  # Should return None
    ]

    for test_name in test_cases:
        result = get_alias_override(test_name)
        status = "✅ FOUND" if result else "❌ NOT FOUND"
        print(f"{status}  '{test_name}' → {result}")

    print("\n" + "="*80)
    print("\nTo use these overrides, modify filename_building_parser.py")
    print("See the instructions in IMPLEMENTATION_GUIDE.md")
    print("="*80 + "\n")
