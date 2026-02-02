#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maintenance data utilities for AskAlfred.

- Normalises categories, statuses and priorities from natural-language queries
- Extracts building names (delegates fully to building_utils)
- Parses maintenance queries into structured components
- Formats maintenance metrics for display

This version removes legacy commented code and inline test harness.
"""

from __future__ import annotations

import logging
import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict
from building_utils import (
    extract_building_from_query as extract_building, resolve_building_name_fuzzy)


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
    "electrical testing": "Electrical Testing",
    "electrical major power loss": "Electrical Major Power Loss",
    "electrical small power loss": "Electrical Small Power Loss",
    "emergency lighting": "Emergency Lighting",
    "extract": "Extract",
    "fire": "Fire alarms",
    "fire doors": "Fire Doors",
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

# Priority mappings
PRIORITY_MAPPINGS = {
    "rm priority 1": "P1",
    "rm priority 2": "P2",
    "rm priority 3": "P3",
    "rm priority 4": "P4",
    "rm priority 5": "P5",
    "rm priority 6": "P6",
    # PPM variants
    "planned & preventative maintenance": "PPM",
    "planned and preventative maintenance": "PPM",
    "ppm": "PPM",

    # Other variants
    "other": "Other",
}

PRIORITY_PATTERN = re.compile(r"rm\s*priority\s*(\d+)", re.IGNORECASE)

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


def normalise_priority(label: str | None) -> Optional[str]:
    """
    Convert raw priority labels (e.g. 'RM Priority 1 - Within 2 hours')
    into canonical codes (e.g. 'P1', 'PPM', 'Other').
    """
    if not label:
        return None
    cleaned = label.strip().lower()

    # Exact match first
    for key, value in PRIORITY_MAPPINGS.items():
        if cleaned.startswith(key) or key in cleaned:
            return value

    # Pattern match for RM Priority X variants
    m = PRIORITY_PATTERN.search(cleaned)
    if m:
        return f"P{m.group(1)}"

    return None


def is_request_metrics(metrics: Dict[str, Any]) -> bool:
    # Heuristic: category -> priority -> status -> int
    for _, prios in (metrics or {}).items():
        if not isinstance(prios, dict) or not prios:
            continue
        for _, statuses in prios.items():
            if not isinstance(statuses, dict) or not statuses:
                continue
            for v in statuses.values():
                try:
                    if int(v) > 0:
                        return True
                except (TypeError, ValueError):
                    pass
    return False

# -----------------------------------------------------------------------------
# Building name extraction
# -----------------------------------------------------------------------------


def extract_building_name_from_query(query: str, known_buildings: List[str]) -> Optional[str]:
    """
    Delegates building extraction entirely to building_utils then fallback.
    known_buildings is passed through when available.
    """
    try:
        for use_cache in (True, False):
            result = extract_building(
                query, known_buildings=known_buildings, use_cache=use_cache)
            if not result:
                continue

            cleaned = result.strip().rstrip("?.!,")
            resolved = resolve_building_name_fuzzy(cleaned)

            if resolved and resolved != cleaned:
                logging.info(
                    "[MaintenanceParser] %s matched: '%s' -> '%s'",
                    "cached" if use_cache else "fallback",
                    cleaned,
                    resolved,
                )
            else:
                logging.info(
                    "[MaintenanceParser] %s matched: '%s'",
                    "cached" if use_cache else "fallback",
                    cleaned,
                )

            return resolved or cleaned

    except Exception:
        logging.exception("Building extraction failed")

    return None

# -----------------------------------------------------------------------------
# Query parsing + formatting
# -----------------------------------------------------------------------------


def parse_priority_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """
    From a label like 'RM Priority 3 - Within 1 working week'
    return ('P3', 'Within 1 working week').

    If label is 'Other' -> (None, None)
    """
    if not label:
        return (None, None)

    s = " ".join(label.split())  # normalise whitespace

    m = re.search(r"(?:RM\s*)?Priority\s*(\d+)", s, re.IGNORECASE)
    pcode = f"P{m.group(1)}" if m else None

    # SLA: take text after the last '-' if it looks like an SLA phrase
    # (keeps it flexible for variants)
    parts = [p.strip() for p in s.split("-")]
    sla = parts[-1] if len(parts) >= 2 else None

    # Heuristic: only treat it as SLA if it contains time-ish keywords
    if sla and not re.search(r"\b(hour|hours|day|days|week|weeks|month|months|maintenance)\b", sla, re.IGNORECASE):
        sla = None

    return (pcode, sla)

# -----------------------------------------------------------------------------
# AGGREGATION FUNCTIONS
# -----------------------------------------------------------------------------


def aggregate_request_metrics(
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate maintenance request metrics.

    Input shape:
      category -> priority_label -> status -> count

    Output:
      {
        "total": int,
        "by_status": {status: count},
        "by_priority_label": {priority: count},
        "by_priority_code": {priority code: count},
        "by_sla_bucket": {sla bucket: count},
      }
    """
    by_status = defaultdict(int)
    by_priority_label = defaultdict(int)
    by_priority_code = defaultdict(int)
    by_sla_bucket = defaultdict(int)
    total = 0

    for _, prios in (metrics or {}).items():
        if not isinstance(prios, dict):
            continue
        for priority_label, statuses in prios.items():
            if not isinstance(statuses, dict):
                continue
            pcode, sla = parse_priority_label(priority_label)
            for status, count in statuses.items():
                try:
                    n = int(count)
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue
                total += n
                by_status[status] += n
                by_priority_label[priority_label] += n
                if pcode:
                    by_priority_code[pcode] += n
                if sla:
                    by_sla_bucket[sla] += n

    return {
        "total": total,
        "by_status": dict(by_status),
        "by_priority_label": dict(by_priority_label),
        "by_priority_code": dict(by_priority_code),
        "by_sla_bucket": dict(by_sla_bucket),
    }


def aggregate_job_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate maintenance job metrics.

    Input shape:
      category -> status -> count

    Output:
      {
        "type": "jobs",
        "total": int,
        "by_status": {status: count},
        "by_category": {category: {"total": int, "by_status": {status: count}}},
      }
    """
    by_status = defaultdict(int)
    by_category: Dict[str, Any] = {}
    total = 0

    for category, statuses in (metrics or {}).items():
        if not isinstance(statuses, dict) or not statuses:
            continue

        cat_total = 0
        cat_by_status = defaultdict(int)

        for status, count in statuses.items():
            try:
                n = int(count)
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue

            total += n
            cat_total += n
            by_status[status] += n
            cat_by_status[status] += n

        by_category[category] = {"total": cat_total,
                                 "by_status": dict(cat_by_status)}

    return {"type": "jobs", "total": total, "by_status": dict(by_status), "by_category": by_category}


def aggregate_request_metrics_by_category(metrics: dict) -> dict:
    """
    Aggregate maintenance request metrics PER CATEGORY.

    Input shape:
      category -> priority_label -> status -> count

    Output:
      {
        "type": "requests",
        "total": int,
        "by_category": {
          category: {
            "total": int,
            "by_status": {status: count},
            "by_priority_label": {priority_label: count},
            "by_priority_code": {priority_code: count},
            "by_sla_bucket": {sla_bucket: count},
          }
        }
      }
    """
    out = {}
    grand_total = 0

    for category, prios in (metrics or {}).items():
        if not isinstance(prios, dict) or not prios:
            continue
        by_status = defaultdict(int)
        by_priority_label = defaultdict(int)
        by_priority_code = defaultdict(int)
        by_sla_bucket = defaultdict(int)
        total = 0

        for priority_label, statuses in prios.items():
            if not isinstance(statuses, dict):
                continue
            pcode, sla = parse_priority_label(priority_label)

            for status, count in statuses.items():
                try:
                    n = int(count)
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue

                total += n
                by_status[status] += n
                by_priority_label[priority_label] += n
                if pcode:
                    by_priority_code[pcode] += n
                if sla:
                    by_sla_bucket[sla] += n

        out[category] = {
            "total": total,
            "by_status": dict(by_status),
            "by_priority_label": dict(by_priority_label),
            "by_priority_code": dict(by_priority_code),
            "by_sla_bucket": dict(by_sla_bucket),
        }
        grand_total += total

    return {"total": grand_total, "by_category": out}

# ---------------------------
# Router / wrapper
# ---------------------------


def aggregate_maintenance_metrics_any(
    metrics: Dict[str, Any],
    *,
    requests_group_by_category: bool = False,
) -> Dict[str, Any]:
    """
    Detects whether `metrics` looks like request-metrics or job-metrics and routes
    to the appropriate aggregator.

    - If request-shaped:
        - returns either aggregate_request_metrics(...) OR
          aggregate_request_metrics_by_category(...) depending on flag.
    - Else:
        - returns aggregate_job_metrics(...)

    Returns a consistent top-level object with a "type" field: "requests" or "jobs".
    """
    if not metrics or not isinstance(metrics, dict):
        # Safe empty response
        return {"type": "unknown", "total": 0, "by_status": {}, "by_category": {}}

    if is_request_metrics(metrics):
        if requests_group_by_category:
            return aggregate_request_metrics_by_category(metrics)
        return aggregate_request_metrics(metrics)

    return aggregate_job_metrics(metrics)


def parse_maintenance_query(query: str, known_buildings: List[str]) -> Dict[str, Optional[str]]:
    """
    Parse the maintenance query into structured parts.
    Returns dict: building_name, category, status, query_type
    """
    result: Dict[str, Optional[str]] = {
        "building_name": None,
        "category": None,
        "priority": None,
        "status": None,
        "query_type": None,
    }

    ql = query.lower()

    result["building_name"] = extract_building_name_from_query(
        query, known_buildings)
    result["category"] = normalise_category(query)

    if any(k in ql for k in ("priority", "ppm", "planned", "preventative", "other")):
        result["priority"] = normalise_priority(query)

    result["status"] = normalise_status(query)

    if "job" in ql:
        result["query_type"] = "jobs"
    elif "request" in ql:
        result["query_type"] = "requests"
    else:
        # sensible default for “how many…”, “show me…”, etc.
        result["query_type"] = "requests"

    return result


def _icon(s: str) -> str:
    s = (s or "").strip().lower()
    if s == "complete":
        return "✓"
    if "progress" in s:
        return "◉"
    if "outstanding" in s or "open" in s:
        return "○"
    return "•"


def fmt(n: int) -> str:
    return f"{n:,}"


def format_job_metrics(
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


def format_request_metrics_summary(metrics: Dict[str, Any], building_name: Optional[str] = None) -> str:
    agg = aggregate_request_metrics(metrics)

    lines = []
    if building_name:
        lines.append(f"### 🏢 **{building_name}**")

    total_requests = agg.get('total', 0)
    if not isinstance(total_requests, int):
        total_requests = 0
    lines.append(f"▪ **Total requests:** {fmt(total_requests)}")

    by_status = agg.get("by_status", {})
    if by_status and isinstance(by_status, dict):
        order = ["Complete", "In progress"]
        sorted_statuses = sorted(by_status.keys(), key=lambda s: (
            order.index(s) if s in order else 999, s))
        lines.append(
            "**Status:** " + " | ".join([f"{_icon(s)} {s}: {fmt(by_status[s])}" for s in sorted_statuses]))

    # optional: top 3 priority labels
    by_priority = agg.get("by_priority_label", {})
    if by_priority and isinstance(by_priority, dict):
        top_p = sorted(by_priority.items(), key=lambda kv: -kv[1])[:3]
        lines.append("**Top priorities:** " +
                     " | ".join([f"{p}: {fmt(c)}" for p, c in top_p]))

    return "\n".join(lines)


def _plural(n: int, word: str) -> str:
    return f"{word}{'' if n == 1 else 's'}"


def _display_status(s: str) -> str:
    s2 = (s or "").strip().lower()
    if s2 == "complete":
        return "Complete"
    if "progress" in s2:
        return "In progress"
    if s2 in ("open", "outstanding"):
        return "Open"
    return s2.title() if s2 else "Unknown"


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
            "Biomedical Sciences Building": {"Complete": 1206, "In progress": 360, "Open": 15},
            "Chemistry School":            {"In progress": 410, "Complete": 1156}
        }
    """

    # Header
    lines = [
        f"### 🏢 **Buildings with maintenance {query_type}:** in the last 12 months",
        "",
        f"📊 **{total_buildings} {_plural(total_buildings, 'building')} found**",
        f"🧾 **{fmt(total_records)} total {_plural(total_records, query_type[:-1])}**",
        f"📎 Showing top {min(limit, len(building_stats))}",
        "\n---\n"
    ]

    # Sort & slice
    ranked = sorted(
        building_stats.items(),
        key=lambda kv: -sum(kv[1].values())
    )[:limit]

    # Output each building
    for bname, stats in ranked:
        total = sum(stats.values())
        parts = [f"{_icon(s)} {_display_status(s)}: {fmt(c)}" for s,
                 c in stats.items() if c > 0]

        lines.append(f"#### 🏛️ **{bname}** — **{fmt(total)} total**")
        if parts:
            lines.append(" | ".join(parts))
        lines.append("")  # spacing

    # Remaining buildings
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


def filter_maintenance_buildings(
    matches: list[dict],
    building: str | None,
    category: str | None,
    priority: str | None,
    status: str | None,
) -> list[dict]:
    """
    Filter building-level maintenance vectors by:
      • building name
      • category
      • priority (P1-P6, PPM, Other)  [requests only]
      • status (open, complete, in progress, etc.)

    Supports:
      • Requests metrics (3-level): category -> priority -> status -> count
      • Jobs metrics (2-level):     category -> status -> count
    """
    category_l = category.lower().strip() if category else None
    status_l = status.lower().strip() if status else None
    _priority_val = normalise_priority(priority) if priority else None
    priority_norm = _priority_val.strip().lower(
    ) if isinstance(_priority_val, str) else None

    filtered: list[dict] = []

    def _parse_metrics(raw_metrics: Any) -> dict:
        """Parse metrics from JSON string or dict."""
        if isinstance(raw_metrics, str):
            try:
                return json.loads(raw_metrics)
            except Exception:
                return {}
        elif isinstance(raw_metrics, dict):
            return raw_metrics
        else:
            return {}

    for m in matches:
        md = m.get("metadata", {}) or {}
        raw_metrics = md.get("maintenance_metrics", {})

        # Parse metrics (may be JSON string)
        metrics = _parse_metrics(raw_metrics)

        if not isinstance(metrics, dict) or not metrics:
            continue

        # Building filter
        bname = md.get("canonical_building_name") or md.get(
            "building_name") or ""
        if building and building.lower() not in bname.lower():
            continue

        # Detect request-shaped metrics (4-level) vs jobs (2-level)
        is_req = False
        try:
            is_req = is_request_metrics(metrics)
        except Exception:
            is_req = False

        # ----------------------------
        # CATEGORY filter
        # ----------------------------
        if category_l:
            if not any(isinstance(k, str) and k.lower() == category_l for k in metrics.keys()):
                continue

        # ----------------------------
        # PRIORITY filter (requests only)
        # ----------------------------
        if priority_norm:
            if not is_req:
                # priority filter doesn't apply to jobs
                continue
            wanted_code = normalise_priority(priority)  # "P3"
            wanted_code_l = wanted_code.lower() if wanted_code else None

            if not wanted_code_l:
                continue

            def _prio_label_matches(priority_label: str, wanted=wanted_code_l) -> bool:
                pcode, _sla = parse_priority_label(priority_label)
                return (pcode or "").lower() == wanted

            def _has_priority_in_cat(cat_key: str) -> bool:
                prios = metrics.get(cat_key, {})
                if not isinstance(prios, dict):
                    return False
                return any(isinstance(p, str) and _prio_label_matches(p) for p in prios.keys())

            if category_l:
                # find actual category key (preserve original case)
                cat_key = next(
                    (k for k in metrics.keys() if isinstance(
                        k, str) and k.lower() == category_l),
                    None
                )
                if not cat_key or not _has_priority_in_cat(cat_key):
                    continue
            else:
                ok = False
                for cat_key, prios in metrics.items():
                    if not isinstance(cat_key, str) or not isinstance(prios, dict):
                        continue
                    if any(isinstance(p, str) and _prio_label_matches(p) for p in prios.keys()):
                        ok = True
                        break
                if not ok:
                    continue

        # ----------------------------
        # STATUS filter
        # ----------------------------
        if status_l:
            #     if is_req:
            # # Aggregate across full 4-level cube
            # agg = aggregate_request_metrics(metrics) or {}
            # by_status = agg.get("by_status", {}) or {}
            # # Compare case-insensitively
            # count_for_status = 0
            # for st, c in by_status.items():
            #     if isinstance(st, str) and st.lower() == status_l and isinstance(c, int):
            #         count_for_status = c
            #         break
            # if count_for_status <= 0:
            #     continue
            if is_req:
                # If category filter is present, only count statuses within that category.
                if category_l:
                    cat_key = next(
                        (k for k in metrics.keys() if isinstance(
                            k, str) and k.lower() == category_l),
                        None
                    )
                    if not cat_key:
                        continue

                    agg = aggregate_request_metrics_by_category(metrics) or {}
                    cat = (agg.get("by_category", {})
                           or {}).get(cat_key, {}) or {}
                    by_status = cat.get("by_status", {}) or {}
                else:
                    agg = aggregate_request_metrics(metrics) or {}
                    by_status = agg.get("by_status", {}) or {}

                # Compare case-insensitively
                count_for_status = 0
                for st, c in by_status.items():
                    if isinstance(st, str) and st.lower() == status_l and isinstance(c, int):
                        count_for_status = c
                        break
                if count_for_status <= 0:
                    continue

            else:
                # Jobs: category -> status -> count
                total = 0
                if category_l:
                    cat_key = next((k for k in metrics.keys() if isinstance(
                        k, str) and k.lower() == category_l), None)
                    if not cat_key:
                        continue
                    statuses = metrics.get(cat_key, {})
                    if isinstance(statuses, dict):
                        for st, c in statuses.items():
                            if isinstance(st, str) and st.lower() == status_l and isinstance(c, int):
                                total += c
                    if total <= 0:
                        continue
                else:
                    for _, statuses in metrics.items():
                        if not isinstance(statuses, dict):
                            continue
                        for st, c in statuses.items():
                            if isinstance(st, str) and st.lower() == status_l and isinstance(c, int):
                                total += c
                    if total <= 0:
                        continue

        filtered.append(m)

    return filtered
