#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve canonical building names for files using the same filename parsing
and normalisation rules as ingestion, without pulling full app config.

Usage:
  python cli/resolve_buildings.py --path data --property-csv Dim-Property.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Iterable, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from building.normaliser import normalise_building_name
from filename_building_parser import extract_building_from_filename


@dataclass(frozen=True)
class Resolution:
    filename: str
    extracted: str | None
    canonical: str
    confidence: float
    source: str


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in path.rglob("*"):
        if item.is_file():
            yield item


def _load_property_csv(path: Path) -> tuple[list[str], dict[str, str], dict[str, str]]:
    canonical_names: list[str] = []
    name_to_canonical: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prop_name = (row.get("Property name") or "").strip()
            if not prop_name:
                continue
            canonical = prop_name
            canonical_names.append(canonical)

            canonical_norm = normalise_building_name(canonical)
            name_to_canonical[canonical_norm] = canonical
            alias_to_canonical[canonical_norm] = canonical

            for key in ("Property names", "Property alternative names"):
                raw = row.get(key)
                if not raw:
                    continue
                for name in str(raw).split(";"):
                    name = name.strip()
                    if not name:
                        continue
                    norm = normalise_building_name(name)
                    name_to_canonical[norm] = canonical
                    alias_to_canonical[norm] = canonical

            condensed = (row.get("UsrFRACondensedPropertyName") or "").strip()
            if condensed:
                norm = normalise_building_name(condensed)
                name_to_canonical[norm] = canonical
                alias_to_canonical[norm] = canonical

    return canonical_names, name_to_canonical, alias_to_canonical


def _fuzzy_match(name: str, canonicals: list[str], threshold: float) -> tuple[Optional[str], float]:
    if not name or not canonicals:
        return None, 0.0
    name_lower = name.lower()
    canonical_lower = [c.lower() for c in canonicals]
    match = get_close_matches(name_lower, canonical_lower, n=1, cutoff=threshold)
    if not match:
        return None, 0.0
    idx = canonical_lower.index(match[0])
    score = SequenceMatcher(None, name_lower, match[0]).ratio()
    return canonicals[idx], score


def resolve_building(
    filename: str,
    canonicals: list[str],
    name_to_canonical: dict[str, str],
    alias_to_canonical: dict[str, str],
    fuzzy_strong: float,
    fuzzy_weak: float,
) -> Resolution:
    extracted = extract_building_from_filename(filename)
    if extracted:
        extracted_norm = normalise_building_name(extracted)
        if extracted_norm in alias_to_canonical:
            return Resolution(filename, extracted, alias_to_canonical[extracted_norm], 1.0, "filename")
        if extracted_norm in name_to_canonical:
            return Resolution(filename, extracted, name_to_canonical[extracted_norm], 1.0, "filename")

        match, score = _fuzzy_match(extracted, canonicals, fuzzy_strong)
        if match and score >= fuzzy_strong:
            return Resolution(filename, extracted, match, score, "filename")
        match, score = _fuzzy_match(extracted, canonicals, fuzzy_weak)
        if match and score >= fuzzy_weak:
            return Resolution(filename, extracted, match, score, "filename")

    return Resolution(filename, extracted, "Unknown", 0.0, "unknown")


def _print_resolution(res: Resolution) -> None:
    extracted = res.extracted or ""
    print(f"{res.filename}\t{extracted}\t{res.canonical}\t{res.confidence:.2f}\t{res.source}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve building names for files.")
    parser.add_argument("--path", required=True, help="File or directory to scan.")
    parser.add_argument("--property-csv", required=True, help="Path to Dim-Property CSV.")
    parser.add_argument("--fuzzy-strong", type=float, default=0.80, help="Strong fuzzy match threshold.")
    parser.add_argument("--fuzzy-weak", type=float, default=0.70, help="Weak fuzzy match threshold.")
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists():
        raise SystemExit(f"Path not found: {base}")

    csv_path = Path(args.property_csv)
    if not csv_path.exists():
        raise SystemExit(f"Property CSV not found: {csv_path}")

    canonicals, name_to_canonical, alias_to_canonical = _load_property_csv(csv_path)
    output_path = Path(args.output).resolve() if args.output else None
    writer = None
    handle = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        writer.writerow(["file", "extracted", "canonical", "confidence", "source"])

    print("file\textracted\tcanonical\tconfidence\tsource")
    for file_path in _iter_files(base):
        res = resolve_building(
            filename=file_path.name,
            canonicals=canonicals,
            name_to_canonical=name_to_canonical,
            alias_to_canonical=alias_to_canonical,
            fuzzy_strong=args.fuzzy_strong,
            fuzzy_weak=args.fuzzy_weak,
        )
        _print_resolution(res)
        if writer:
            writer.writerow([
                res.filename,
                res.extracted or "",
                res.canonical,
                f"{res.confidence:.2f}",
                res.source,
            ])
    if handle:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
