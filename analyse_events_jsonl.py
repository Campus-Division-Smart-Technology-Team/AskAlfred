#!/usr/bin/env python3
"""
Analyze structured building assignment events exported by ingest.

Usage:
  python analyse_events_jsonl.py building_events.jsonl
  python analyse_events_jsonl.py building_events.jsonl --export analysis.csv
"""

import json
import sys
import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List


def load_events(path: Path):
    """Stream JSONL events safely."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def analyse(events: List[dict]) -> dict:
    """Analyse building assignment quality."""
    results = {
        "total": len(events),
        "unknown": [],
        "flagged": [],
        "low_confidence": [],
        "by_building": Counter(),
        "by_source": Counter(),
    }

    for ev in events:
        file = ev.get("file")
        building = ev.get("canonical_building_name")
        conf = float(ev.get("confidence", 0))
        src = ev.get("source")
        flag = ev.get("flag_review", False)

        results["by_building"][building] += 1
        results["by_source"][src] += 1

        if building == "Unknown":
            results["unknown"].append(ev)

        if flag:
            results["flagged"].append(ev)

        if src == "filename" and conf < 0.75:
            results["low_confidence"].append(ev)
        if src == "text" and conf < 0.70:
            results["low_confidence"].append(ev)

    return results


def print_report(results: dict):
    print("\n=================== BUILDING ASSIGNMENT ANALYSIS ===================\n")

    print(f"Total assignments: {results['total']}")
    print(f"Unknown: {len(results['unknown'])}")
    print(f"Flagged for review: {len(results['flagged'])}")
    print(f"Low confidence: {len(results['low_confidence'])}")

    print("\nTop buildings:")
    for building, count in results["by_building"].most_common(10):
        print(f"  {building}: {count}")

    print("\nBy source:")
    for src, count in results["by_source"].most_common():
        print(f"  {src}: {count}")

    print("\n===================================================================\n")


def export_csv(results: dict, output: str):
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "building", "confidence", "source", "flag_review"])

        for ev_list in (results["unknown"], results["flagged"], results["low_confidence"]):
            for ev in ev_list:
                w.writerow([
                    ev.get("file"),
                    ev.get("canonical_building_name"),
                    ev.get("confidence"),
                    ev.get("source"),
                    ev.get("flag_review"),
                ])

    print(f"✅ Exported CSV: {output}")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python analyse_events_jsonl.py building_events.jsonl [--export file.csv]")
        return

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    export = None
    if "--export" in sys.argv:
        i = sys.argv.index("--export")
        export = sys.argv[i + 1] if i + 1 < len(sys.argv) else None

    events = load_events(path)
    results = analyse(events)
    print_report(results)

    if export:
        export_csv(results, export)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
