#!/usr/bin/env python3
"""
CSV Alias Helper - Identifies missing building name aliases in Property CSV

This script analyzes your filenames and Property CSV to identify which building
name variations are missing, helping you fix "Maintenance" misassignments.

Usage:
    python3 csv_alias_helper.py --csv /path/to/Dim-Property.csv --files /path/to/documents/

Output:
    - List of extracted building names from filenames
    - List of missing aliases in CSV
    - Suggested CSV updates
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd

# Import the filename extraction function
try:
    from filename_building_parser import extract_building_from_filename
except ImportError:
    print("ERROR: Cannot import filename_building_parser module")
    print("Make sure filename_building_parser.py is in the same directory")
    sys.exit(1)


def load_csv_aliases(csv_path: str) -> Tuple[Dict[str, str], Set[str]]:
    """
    Load existing aliases from Property CSV.

    Returns:
        (alias_to_canonical_map, all_aliases_lowercase)
    """
    df = pd.read_csv(csv_path)

    alias_to_canonical = {}
    all_aliases = set()

    for _, row in df.iterrows():
        canonical = row.get("Property name")
        if pd.isna(canonical):
            continue

        canonical = str(canonical).strip()
        canonical_lower = canonical.lower()

        # Add canonical name itself
        all_aliases.add(canonical_lower)
        alias_to_canonical[canonical_lower] = canonical

        # Add property names
        if pd.notna(row.get("Property names")):
            for name in str(row["Property names"]).split(";"):
                name = name.strip().lower()
                if name:
                    all_aliases.add(name)
                    alias_to_canonical[name] = canonical

        # Add alternative names
        if pd.notna(row.get("Property alternative names")):
            for name in str(row["Property alternative names"]).split(";"):
                name = name.strip().lower()
                if name:
                    all_aliases.add(name)
                    alias_to_canonical[name] = canonical

    return alias_to_canonical, all_aliases


def scan_files(directory: str) -> List[Tuple[str, str]]:
    """
    Scan directory for files and extract building names.

    Returns:
        List of (filename, extracted_building_name) tuples
    """
    path = Path(directory)
    results = []

    extensions = ['.pdf', '.docx']

    for ext in extensions:
        for filepath in path.rglob(f'*{ext}'):
            filename = filepath.name
            extracted = extract_building_from_filename(filename)
            if extracted:
                results.append((filename, extracted))

    return results


def find_missing_aliases(
    extracted_names: List[Tuple[str, str]],
    alias_to_canonical: Dict[str, str],
    all_aliases: Set[str]
) -> Dict[str, List[str]]:
    """
    Find extracted building names that are missing from CSV aliases.

    Returns:
        Dict mapping extracted names to list of files using that name
    """
    missing = {}

    for filename, extracted in extracted_names:
        extracted_lower = extracted.lower()

        # Check if this extracted name is in aliases
        if extracted_lower not in all_aliases:
            if extracted not in missing:
                missing[extracted] = []
            missing[extracted].append(filename)

    return missing


def suggest_csv_updates(
    missing: Dict[str, List[str]],
    alias_to_canonical: Dict[str, str]
) -> List[Tuple[str, str, List[str]]]:
    """
    Suggest CSV updates for missing aliases.

    Returns:
        List of (extracted_name, suggested_canonical, files) tuples
    """
    suggestions = []

    for extracted, files in missing.items():
        # Try to find a similar canonical name
        extracted_lower = extracted.lower()

        # Simple fuzzy matching
        best_match = None
        best_score = 0

        for alias_lower, canonical in alias_to_canonical.items():
            # Check for substring match
            if extracted_lower in alias_lower or alias_lower in extracted_lower:
                score = len(set(extracted_lower.split())
                            & set(alias_lower.split()))
                if score > best_score:
                    best_score = score
                    best_match = canonical

        suggestions.append((extracted, best_match or "UNKNOWN", files))

    return suggestions


def print_report(
    extracted_names: List[Tuple[str, str]],
    missing: Dict[str, List[str]],
    suggestions: List[Tuple[str, str, List[str]]]
):
    """Print a formatted report."""

    print("\n" + "="*80)
    print("CSV ALIAS ANALYSIS REPORT")
    print("="*80 + "\n")

    # Summary
    total_files = len(extracted_names)
    total_unique = len(set(name for _, name in extracted_names))
    total_missing = len(missing)

    print(f"📊 SUMMARY:")
    print(f"   Total files scanned: {total_files}")
    print(f"   Unique building names extracted: {total_unique}")
    print(f"   Missing from CSV aliases: {total_missing}")
    print()

    if not missing:
        print("✅ Great! All extracted building names are in your CSV aliases.")
        print("   No action needed.")
        return

    # Missing aliases
    print("❌ MISSING ALIASES:")
    print("-" * 80)
    for extracted, files in missing.items():
        print(f"\n   Building name: \"{extracted}\"")
        print(f"   Found in {len(files)} file(s):")
        for f in files[:3]:  # Show first 3 files
            print(f"      - {f}")
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more")
    print()

    # Suggestions
    print("\n📝 SUGGESTED CSV UPDATES:")
    print("-" * 80)
    print("\nAdd these aliases to your Property CSV:\n")

    for extracted, suggested_canonical, files in suggestions:
        if suggested_canonical != "UNKNOWN":
            print(f"Property name: {suggested_canonical}")
            print(f"  → Add to 'Property alternative names': \"{extracted}\"")
            print(f"     (affects {len(files)} file(s))")
            print()
        else:
            print(
                f"⚠️  Cannot auto-suggest canonical name for: \"{extracted}\"")
            print(f"   Please manually find the correct building in your CSV")
            print(f"   (affects {len(files)} file(s))")
            print()

    # CSV format example
    print("\n📋 CSV FORMAT EXAMPLE:")
    print("-" * 80)
    print("\nProperty name,Property alternative names")
    for extracted, suggested_canonical, files in suggestions[:3]:
        if suggested_canonical != "UNKNOWN":
            print(f"{suggested_canonical},\"[existing aliases];{extracted}\"")
    print("\nRemember: Use semicolons (;) to separate multiple aliases")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Identify missing building name aliases in Property CSV"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Property CSV file (e.g., Dim-Property.csv)"
    )
    parser.add_argument(
        "--files",
        required=True,
        help="Path to directory containing documents"
    )
    parser.add_argument(
        "--export",
        help="Export missing aliases to CSV file"
    )

    args = parser.parse_args()

    # Validate paths
    csv_path = Path(args.csv)
    files_path = Path(args.files)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    if not files_path.exists():
        print(f"ERROR: Directory not found: {args.files}")
        sys.exit(1)

    print("Loading Property CSV...")
    alias_to_canonical, all_aliases = load_csv_aliases(args.csv)
    print(f"   Loaded {len(all_aliases)} aliases from CSV")

    print("\nScanning files for building names...")
    extracted_names = scan_files(args.files)
    print(f"   Extracted building names from {len(extracted_names)} files")

    print("\nAnalyzing missing aliases...")
    missing = find_missing_aliases(
        extracted_names, alias_to_canonical, all_aliases)

    suggestions = suggest_csv_updates(missing, alias_to_canonical)

    # Print report
    print_report(extracted_names, missing, suggestions)

    # Export if requested
    if args.export and missing:
        export_df = pd.DataFrame([
            {
                "Extracted Name": extracted,
                "Suggested Canonical": canonical,
                "File Count": len(files),
                "Example Files": "; ".join(files[:3])
            }
            for extracted, canonical, files in suggestions
        ])
        export_df.to_csv(args.export, index=False)
        print(f"\n✅ Exported missing aliases to: {args.export}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Update your Property CSV with the suggested aliases")
    print("2. Re-run batch_ingest.py with --clear-cache")
    print("3. Verify in logs that buildings are assigned correctly")
    print("\nFor detailed instructions, see: MAINTENANCE_ASSIGNMENT_DIAGNOSTIC.md\n")


if __name__ == "__main__":
    main()
