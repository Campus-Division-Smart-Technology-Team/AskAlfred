#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maintenance data utilities for AskAlfred.

- Normalises categories and statuses from natural-language queries
- Extracts building names (delegates fully to building_utils)
- Parses maintenance queries into structured components
- Formats maintenance metrics for display

This version removes legacy commented code and inline test harness.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Any, Union, FrozenSet
from difflib import get_close_matches

# Prefer the building_utils extractor (it uses cache + n-gram fallback internally)
from building_utils import extract_building_from_query as extract_building
from buildings_cache import BUILDING_NAMES_CACHE


# Common false positives to filter out when detecting building names
INVALID_BUILDING_NAMES: FrozenSet[str] = frozenset(
    {
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

# Category mappings
MAINTENANCE_CATEGORIES = {
    "asbestos": "Asbestos",
    "asu": "ASU",
    "bems": "BEMS Controls",
    "controls": "BEMS Controls",
    "ceiling": "Ceilings",
    "ceilings": "Ceilings",
    "cmi": "CMI PPM",
    "ppm": "CMI PPM",
    "cold water": "Cold Water Systems",
    "compressed air": "Compressed air",
    "cooker": "Cookers",
    "cookers": "Cookers",
    "cooling": "Cooling",
    "delivery": "Deliveries",
    "deliveries": "Deliveries",
    "door": "Doors",
    "doors": "Doors",
    "drainage": "Drainage",
    "electrical": "Electrical",
    "emergency": "Emergency",
    "extract": "Extract",
    "fire": "Fire alarms",
    "alarm": "Fire alarms",
    "floor": "Floors",
    "floors": "Floors",
    "heating": "Heating",
    "hot water": "Hot Water Systems",
    "lighting": "Lighting",
    "plumbing": "Plumbing",
    "roof": "Roofs",
    "roofs": "Roofs",
    "security": "Security",
    "ventilation": "Ventilation",
    "wall": "Walls",
    "walls": "Walls",
    "window": "Windows",
    "windows": "Windows",
}

# Status mappings
STATUS_MAPPINGS = {
    "complete": "Complete",
    "completed": "Complete",
    "done": "Complete",
    "finished": "Complete",
    "in progress": "In progress",
    "progress": "In progress",
    "ongoing": "In progress",
    "active": "In progress",
    "pending": "In progress",
}


# -----------------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------------

def normalise_category(query_text: str) -> Optional[str]:
    """
    Normalise a maintenance category from user text.
    """
    q = query_text.lower()
    for key, category in MAINTENANCE_CATEGORIES.items():
        if key in q:
            return category
    return None


def normalise_status(query_text: str) -> Optional[str]:
    """
    Normalise a maintenance status from user text.
    """
    q = query_text.lower()
    for key, status in STATUS_MAPPINGS.items():
        if key in q:
            return status
    return None


# -----------------------------------------------------------------------------
# Building name extraction
# -----------------------------------------------------------------------------

def extract_building_name_from_query(query: str, known_buildings: List[str]) -> Optional[str]:
    """
    Delegates building extraction entirely to building_utils then fallback.
    known_buildings is passed through when available.
    """
    try:
        result = extract_building(
            query, known_buildings=known_buildings, use_cache=True)
        if result:
            logging.info(
                "[MaintenanceParser] building_utils matched: '%s'", result)
            return result
        # As a fallback, disable cache to let n-gram + fuzzy kick in
        result = extract_building(
            query, known_buildings=known_buildings, use_cache=False)
        if result:
            logging.info(
                "[MaintenanceParser] fallback building_utils matched: '%s'", result)
            return result
    except Exception as e:
        logging.error("Building extraction failed: %s", e)
        return None
    return None

# -----------------------------------------------------------------------------
# Query parsing + formatting
# -----------------------------------------------------------------------------


def parse_maintenance_query(query: str, known_buildings: List[str]) -> Dict[str, Optional[str]]:
    """
    Parse the maintenance query into structured parts.
    Returns dict: building_name, category, status, query_type
    """
    result: Dict[str, Optional[str]] = {
        "building_name": None,
        "category": None,
        "status": None,
        "query_type": None,
    }

    result["building_name"] = extract_building_name_from_query(
        query, known_buildings)
    result["category"] = normalise_category(query)
    result["status"] = normalise_status(query)

    ql = query.lower()
    if "request" in ql:
        result["query_type"] = "requests"
    elif "job" in ql:
        result["query_type"] = "jobs"

    return result


def _icon(s: str) -> str:
    s = s.lower()
    if s == "complete":
        return "✓"
    if "progress" in s:
        return "◉"
    if "outstanding" in s or "open" in s:
        return "○"
    return "•"


def fmt(n: int) -> str:
    return f"{n:,}"


def format_maintenance_metrics(
    metrics: Dict[str, Dict[str, int]],
    category_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    return_dict: bool = False,
    building_name: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:

    if not metrics:
        return "No maintenance data available."

    # No category filter -> summary per building
    if category_filter is None:
        status_totals: Dict[str, int] = {}

        for statuses in metrics.values():
            for status, count in (statuses or {}).items():
                if status_filter and status != status_filter:
                    continue
                status_totals[status] = status_totals.get(status, 0) + count

        if not status_totals:
            return "No matching maintenance data found."

        order = ["Complete", "In progress"]
        sorted_statuses = sorted(
            status_totals, key=lambda s: (
                order.index(s) if s in order else 999, s)
        )

        total = sum(status_totals.values())

        if return_dict:
            return {"statuses": {s: status_totals[s] for s in sorted_statuses}, "total": total}

        lines = []
        if building_name:
            lines.append(f"### 🏢 **{building_name}**")

        lines.append("**Status Summary:**")

        for s in sorted_statuses:
            lines.append(f"&nbsp;&nbsp;{_icon(s)} **{s}:** {status_totals[s]}")

        lines.append(f"▪ **Total:** {total}")

        return "\n".join(lines)

    # ✅ Category filter → show only that category
    cat = category_filter
    statuses = metrics.get(cat, {})

    if not statuses:
        return f"No maintenance data for category **{cat}**."

    order = ["Complete", "In progress"]
    sorted_statuses = sorted(
        statuses, key=lambda s: (
            order.index(s) if s in order else 999, s
        )
    )
    total = sum(statuses.values())

    lines = []
    if building_name:
        lines.append(f"### 🏢 **{building_name}**")
    lines.append(f"**Category:** {cat}")

    for s in sorted_statuses:
        lines.append(f"&nbsp;&nbsp;**{s}:** {statuses[s]}")

    lines.append(f"▪ **Total:** {fmt(total)}")

    return "\n".join(lines)


def _plural(n: int, word: str) -> str:
    return f"{word}{'' if n == 1 else 's'}"


def format_multi_building_metrics(
    building_stats: Dict[str, Dict[str, int]],
    total_buildings: int,
    total_records: int,
    query_type: str,
    limit: int = 10,
) -> str:
    """
    Format grouped maintenance metrics for multiple buildings
    building_stats sample:
        {
            "Biomedical Sciences Building": {"complete": 1206, "in progress": 360, "open": 15},
            "Chemistry School":            {"in progress": 410, "complete": 1156}
        }
    """

    # ✅ Header
    lines = [
        f"### 🏢 **Buildings with maintenance {query_type}:** in the last 12 months",
        "",
        f"📊 **{total_buildings} {_plural(total_buildings, 'building')} found**",
        f"🧾 **{fmt(total_records)} total {_plural(total_records, query_type[:-1])}**",
        f"📎 Showing top {min(limit, len(building_stats))}",
        "\n---\n"
    ]

    # ✅ Sort & slice
    ranked = sorted(
        building_stats.items(),
        key=lambda kv: -sum(kv[1].values())
    )[:limit]

    # ✅ Output each building
    for bname, stats in ranked:
        total = sum(stats.values())
        parts = [f"{_icon(s)} {s.title()}: {fmt(c)}" for s,
                 c in stats.items() if c > 0]

        lines.append(f"#### 🏛️ **{bname}** — **{fmt(total)} total**")
        if parts:
            lines.append(" | ".join(parts))
        lines.append("")  # spacing

    # ✅ Remaining buildings
    remaining = total_buildings - len(ranked)
    if remaining > 0:
        lines.append(
            f"➕ **{fmt(remaining)} more {_plural(remaining, 'building')}** not shown…")

    return "\n".join(lines)


def calculate_maintenance_summary(metrics: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """
    Return totals and completion rate from metrics.
    """
    total = 0
    total_complete = 0
    total_in_progress = 0
    for _, statuses in (metrics or {}).items():
        for status, count in (statuses or {}).items():
            total += count
            if status.lower() == "complete":
                total_complete += count
            elif "progress" in status.lower():
                total_in_progress += count
    return {
        "total_items": total,
        "total_complete": total_complete,
        "total_in_progress": total_in_progress,
        "categories_count": len(metrics or {}),
        "completion_rate": round((total_complete / total) if total else 0.0, 3),
    }
