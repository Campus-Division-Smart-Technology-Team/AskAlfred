#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counting, ranking and maintenance query routing for AskAlfred (ChatGPT test version)

✅ Full production logic with Pinecone access
✅ Uses maintenance_utils_chatgpt for parsing and formatting
✅ Logging at INFO level for query tracing
"""

from __future__ import annotations
from typing import Any, Dict, List, cast

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union, FrozenSet, DefaultDict, TypedDict
from collections import defaultdict
from pinecone_utils import open_index, normalise_matches
from config import TARGET_INDEXES, resolve_namespace, normalise_ns, get_display_namespace, get_internal_namespace
from building_utils import (BuildingCacheManager,
                            normalise_building_name,
                            populate_building_cache_from_multiple_indexes,
                            extract_building_from_query)

# ---------------------------------------------------------------------
# ---- Maintenance typing helpers ----
# ---------------------------------------------------------------------


class MetricsDict(TypedDict):
    building_name: str
    maintenance_metrics: defaultdict[str, defaultdict[str, int]]


def make_metrics_dict() -> MetricsDict:
    return {
        "building_name": "",
        "maintenance_metrics": defaultdict(lambda: defaultdict(int))
    }

# -----------------------------------------------------------------------------
# Constants that need to be available before imports
# -----------------------------------------------------------------------------


PROPERTY_CONDITION_NORMALISATION = {
    "condition a": "Condition A",
    "condition b": "Condition B",
    "condition c": "Condition C",
    "condition d": "Condition D",
    "derelict": "DERELICT",
    "no maintenance responsibility": "No Maintenance Responsibility",
}


# -----------------------------------------------------------------------------
# Import maintenance_utils with fallback stubs
# -----------------------------------------------------------------------------

INVALID_BUILDING_NAMES: FrozenSet[str]
try:
    from maintenance_utils import (
        parse_maintenance_query,
        INVALID_BUILDING_NAMES, _plural,
        format_multi_building_metrics,
    )
    MAINTENANCE_UTILS_AVAILABLE = True
    logging.info("✅ maintenance_utils successfully imported.")

except Exception as exc:
    logging.warning("maintenance_utils_chatgpt import failed: %s", exc)
    MAINTENANCE_UTILS_AVAILABLE = False

    # --- Stub functions to prevent 'possibly unbound' warnings ---
    def parse_maintenance_query(query: str, known_buildings: list[str]) -> dict[str, Optional[str]]:
        return {"building_name": None, "category": None, "status": None, "query_type": None}

    def normalise_category(query_text: str) -> Optional[str]:
        return None

    def normalise_status(query_text: str) -> Optional[str]:
        return None

    def format_maintenance_metrics(
        metrics: dict[str, dict[str, int]],
        category_filter: str | None = None,
        status_filter: str | None = None,
        return_dict: bool = False
    ) -> Union[str, Dict[str, Any]]:
        return "Maintenance metrics formatting not available."

    # Stub for INVALID_BUILDING_NAMES if import fails
    INVALID_BUILDING_NAMES: FrozenSet[str] = frozenset()

    def _plural(n: int, word: str) -> str:
        return f"{word}{'' if n == 1 else 's'}"


# -----------------------------------------------------------------------------
# Initialise logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logging.info("✅ counting_queries.py (FULL FUNCTIONAL) loaded")

# -----------------------------------------------------------------------------
# Building Cache Initialisation in builing_utils.py
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Constants & Regex Patterns
# -----------------------------------------------------------------------------
COUNTING_PATTERNS = [
    re.compile(r"\bhow\s+many\s+buildings?\b", re.IGNORECASE),
    re.compile(r"\bcount\s+(?:the\s+)?buildings?\b", re.IGNORECASE),
    re.compile(r"\bnumber\s+of\s+buildings?\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\s+\w+\s+(?:have|with|contain)\b", re.IGNORECASE),
    re.compile(r"\blist\s+all\s+buildings?\b", re.IGNORECASE),
    # Exclude property condition and derelict queries from counting
    re.compile(
        r"\bwhich\s+buildings?\s+(?:have|are)\b(?!\s+(?:derelict|in\s+condition|condition\s+[a-d]))",
        re.IGNORECASE
    ),

]

MAINTENANCE_PATTERNS = [
    re.compile(r"\bmaintenance\b", re.IGNORECASE),
    re.compile(r"\b(request|job)s?\b", re.IGNORECASE),
    re.compile(r"\bwork\s+orders?\b", re.IGNORECASE),
    re.compile(r"\bplanned\s+maintenance\b", re.IGNORECASE),
    re.compile(r"\bppm\b", re.IGNORECASE),
]

RANKING_PATTERNS = [
    # direct rank command
    re.compile(r"\brank(?:ing)?\s+buildings?\b", re.IGNORECASE),
    # comparative statements about size
    re.compile(
        r"\b(?:largest|biggest|smallest|highest|lowest|tallest)\b", re.IGNORECASE),
    # “top N buildings”
    re.compile(r"\btop\s+\d+\s+buildings?\b", re.IGNORECASE),
    # “top buildings” (no number)
    re.compile(r"\btop\s+buildings?\b", re.IGNORECASE),
    # “sort buildings by …”
    re.compile(r"\bsort\s+buildings?\s+by\b", re.IGNORECASE),
    # “buildings with the biggest area”
    re.compile(
        r"\bbuildings?\s+(?:with|by|having)\s+(?:the\s+)?(?:largest|biggest|smallest)\b", re.IGNORECASE),
    # “by area / by size / gross / net / sqm / square metres”
    re.compile(
        r"\bby\s+(?:area|size|gross|net|sqm|square\s+metre|square\s+meter)\b", re.IGNORECASE),
    # “compare building sizes”
    re.compile(r"\bcompare\s+buildings?\b", re.IGNORECASE),
]


MAINTENANCE_KEYWORDS = [
    "asbestos", "asu", "bems", "ceiling", "cold water", "cooker", "cooling",
    "drainage", "electrical", "heating", "lighting", "plumbing", "roof",
    "ventilation", "window", "door", "floor", "request", "job", "maintenance",
]

DOC_TYPE_MAPPINGS = {
    "fra": "fire_risk_assessment",
    "fire risk": "fire_risk_assessment",
    "fire assessment": "fire_risk_assessment",
    "bms": "operational_doc",
    "building management": "operational_doc",
    "operational": "operational_doc",
    "o&m": "operational_doc",
    "planon": "planon_data",
    "property": "planon_data",
    "request": "maintenance_request",
    "job": "maintenance_job",
    "requests": "maintenance_request",
    "request": "maintenance_request",
    "jobs": "maintenance_job",
    "job": "maintenance_job",
}

DOC_TYPE_NAMES_SIMPLE = {
    "fire_risk_assessment": "Fire Risk Assessments",
    "operational_doc": "BMS/Operational",
    "planon_data": "Planon property data",
    "maintenance_request": "Maintenance Requests",
    "maintenance_job": "Maintenance Jobs",
    "unknown": "Unknown document types",
}

PRIORITY_MAPPING = {
    "rm priority 1": "P1",
    "rm priority 2": "P2",
    "rm priority 3": "P3",
    "rm priority 4": "P4",
    "rm priority 5": "P5",
    "rm priority 6": "P6",
    "planned & preventative maintenance": "PPM",
    "other": "Other",
}


PRIORITY_PATTERN = re.compile(r"rm\s*priority\s*(\d+)", re.IGNORECASE)


def normalise_priority(label: str) -> str:
    """
    Convert raw priority labels (e.g. 'RM Priority 1 - Within 2 hours')
    into canonical codes (e.g. 'P1', 'PPM', 'Other').
    """
    if not label:
        return "Unknown"

    cleaned = label.strip().lower()

    # Exact match first
    for key, value in PRIORITY_MAPPING.items():
        if cleaned.startswith(key):
            return value

    # Pattern match for RM Priority X variants
    m = PRIORITY_PATTERN.search(label)
    if m:
        return f"P{m.group(1)}"

    return "Other" if cleaned == "other" else cleaned


DEFAULT_BATCH_TOP_K = 1000
MAX_SAFE_TOP_K = 10000

# -----------------------------------------------------------------------------
# Detection helpers
# -----------------------------------------------------------------------------


def is_counting_query(query: str) -> bool:
    q = query.lower().strip()
    return any(p.search(q) for p in COUNTING_PATTERNS)


def is_maintenance_query(query: str) -> bool:
    if not MAINTENANCE_UTILS_AVAILABLE:
        return False
    q = query.lower().strip()
    if any(p.search(q) for p in MAINTENANCE_PATTERNS):
        return True
    return any(k in q for k in MAINTENANCE_KEYWORDS) and (
        "request" in q or "job" in q or "maintenance" in q
    )


def is_property_condition_query(query: str) -> bool:
    q = query.lower()
    keywords = list(PROPERTY_CONDITION_NORMALISATION.keys()) + \
        ["derelict", "condition", "property condition"]
    return any(k in q for k in keywords)


def is_ranking_query(query: str) -> bool:
    q = query.lower().strip()
    return any(p.search(q) for p in RANKING_PATTERNS)


def extract_document_type_from_query(query: str) -> Optional[str]:
    q = query.lower()
    for term, doc_type in DOC_TYPE_MAPPINGS.items():
        if term in q:
            return doc_type
    return None


def get_maintenance_heading(query_type: str) -> str:
    """Heading based on maintenance query type."""
    if query_type == "requests":
        return "Maintenance Requests for Last 12 months"
    if query_type == "jobs":
        return "Maintenance Jobs for Last 12 months"
    return "Maintenance Jobs and Requests for Last 12 months"

# -----------------------------------------------------------------------------
# Property Condition Query Helpers
# -----------------------------------------------------------------------------


def normalise_property_condition(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    for k, v in PROPERTY_CONDITION_NORMALISATION.items():
        if k in q or v.lower() in q:
            return v
    return None


def query_property_by_condition(condition: str, index_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Query buildings by property condition from the 'planon_data' namespace.
    """
    if index_names is None:
        index_names = TARGET_INDEXES

    namespace = 'planon_data'
    canonical_condition = condition
    matching_buildings: List[Dict[str, Any]] = []

    for idx_name in index_names:
        matches = _query_index_with_batches(
            idx_name=idx_name,
            namespace=namespace,
            filter_dict=None,
            top_k=DEFAULT_BATCH_TOP_K,
        )

        for match in matches:
            metadata = match.get('metadata', {}) or {}
            prop_condition = metadata.get('Property condition', '')
            if isinstance(prop_condition, str) and prop_condition.strip() == canonical_condition:
                matching_buildings.append({
                    'building_name': metadata.get('canonical_building_name'),
                    'condition': prop_condition,
                    'postcode': metadata.get('Property postcode', ''),
                    'campus': metadata.get('Property campus', ''),
                    'gross_area': metadata.get('Property gross area (sq m)', ''),
                    'net_area': metadata.get('Property net area (sq m)', '')
                })

    return {
        'condition_filter': condition,
        'matching_buildings': matching_buildings,
        'building_count': len(matching_buildings),
        'namespace': namespace
    }


def rank_buildings_by_area(
    area_type: str = "gross",
    order: str = "desc",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Rank buildings by gross or net area from Planon data vectors in Pinecone.

    Args:
        area_type: "gross" or "net" (default: gross)
        order: "asc" or "desc" (default: desc)
        limit: max number of buildings to return (default: None → all)

    Returns:
        Dict structured for generate_ranking_answer().
    """
    logging.info("📊 rank_buildings_by_area() called - area_type='%s', order='%s', limit=%s",
                 area_type, order, limit)

    # Which metadata field to read
    area_key = (
        'Property gross area (sq m)' if area_type == 'gross'
        else 'Property net area (sq m)'
    )
    logging.info("   Using metadata field: '%s'", area_key)

    namespace = "planon_data"
    buildings: Dict[str, Dict[str, Any]] = {}

    # Fetch all vectors from each target index
    logging.info("   Querying %d target indexes for namespace '%s'",
                 len(TARGET_INDEXES), namespace)
    for idx in TARGET_INDEXES:
        matches = _query_index_with_batches(
            idx_name=idx,
            namespace=namespace,
            filter_dict=None,
            top_k=DEFAULT_BATCH_TOP_K
        )
        logging.info("   Index '%s': Retrieved %d matches", idx, len(matches))

        for match in matches:
            md = match.get("metadata", {}) or {}

            # Get canonical building name if available
            bname = md.get("canonical_building_name")
            if not bname:
                continue

            # Area extraction and numeric conversion
            area_val = md.get(area_key)
            try:
                area_val = float(area_val) if area_val not in (
                    None, "", "N/A") else None
            except Exception:
                area_val = None

            if area_val is None or area_val <= 0:
                continue

            # Deduplicate by building name — keep largest reported area
            if bname not in buildings or area_val > buildings[bname]["area"]:
                buildings[bname] = {
                    "building_name": bname,
                    "area": area_val,
                    "metadata": md
                }

    total = len(buildings)
    logging.info("   Total buildings with valid area data: %d", total)
    if total == 0:
        logging.warning(
            "   ⚠️ No buildings found with area data - returning empty results")
        return {
            "ranking_type": "area",
            "metric": f"{area_type}_area_sq_m",
            "order": order,
            "total_buildings": 0,
            "results": []
        }

    # Sort by area
    reverse = (order == "desc")
    ranked = sorted(buildings.values(),
                    key=lambda x: x["area"], reverse=reverse)
    logging.info("   Buildings sorted by area (%s order)",
                 "descending" if reverse else "ascending")

    # Apply limit if given
    if limit is not None:
        original_count = len(ranked)
        ranked = ranked[:limit]
        logging.info(
            "   Applied limit: %d (reduced from %d buildings)", limit, original_count)
    else:
        logging.info(
            "   No limit applied; returning all %d buildings", len(ranked))

    # Add rank numbers
    for i, item in enumerate(ranked, 1):
        item["rank"] = i

    # Build results in expected format for generate_ranking_answer
    results = [
        {
            "building_name": item["building_name"],
            "value": item["area"],  # expected key for metric value
            "rank": item["rank"],
            "metadata": item["metadata"]
        }
        for item in ranked
    ]
    logging.info(
        "✅ rank_buildings_by_area() completed - returning %d ranked buildings", len(results))

    return {
        "ranking_type": "area",
        "metric": f"{area_type}_area_sq_m",
        "order": order,
        "total_buildings": total,
        "results": results
    }


# -----------------------------------------------------------------------------
# Pinecone Query Helpers
# -----------------------------------------------------------------------------


def _dedupe_matches_by_key(matches):
    """
    Generic dedupe for document search.
    NOT used for maintenance vectors because those are already 1-per-building.
    """
    deduped = []
    seen = set()

    for m in matches:
        md = m.get("metadata", {}) or {}

        key = (
            md.get("key") or
            hash(str(md))
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(m)

    return deduped


def _query_index_with_batches(
    idx_name: str,
    namespace: Optional[str],
    filter_dict: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_BATCH_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Query a Pinecone index in batches with optional filters.
    Returns all matches from all batches combined and deduplicated.
    """
    from pinecone_utils import query_all_chunks

    namespace = normalise_ns(namespace)
    display_namespace = get_display_namespace(namespace)

    try:
        idx = open_index(idx_name)
        stats = idx.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        total_vecs = ns_stats.get("vector_count", 0)

        if total_vecs == 0:
            print(
                f"DEBUG: namespace='{namespace}' raw_stats_keys={stats.get('namespaces', {}).keys()}")
            logging.info(
                f"[{idx_name}] namespace={namespace} has 0 vectors")
            return []

        safe_top_k = min(top_k, MAX_SAFE_TOP_K)
        logging.info(
            f"[{idx_name}] Fetching up to {safe_top_k} vectors from namespace={display_namespace}")
        # Use zero vector to fetch all
        dim = idx.describe_index_stats().get("dimension", 1536)
        zero_vec = [0.0] * dim

        all_matches = query_all_chunks(
            index=idx,
            namespace=namespace,
            query_vector=zero_vec,
            filter_dict=filter_dict,
            top_k=safe_top_k,
            include_metadata=True,
        )

        # normalised = normalise_matches(all_matches)
        deduped = _dedupe_matches_by_key(all_matches)
        logging.info(
            f"[{idx_name}] Retrieved {len(deduped)} unique matches (from {len(all_matches)} total)")

        return deduped

    except Exception as e:
        logging.error(f"Error querying {idx_name}: {e}")
        return []


def _query_index_with_fallback(
    idx_name: str,
    primary_namespace: Optional[str],
    filter_dict: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_BATCH_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Query index with fallback to alternative namespaces if primary returns no results.

    This handles cases where:
    - Namespace mapping is incorrect
    - Data was ingested with different namespace than expected
    - Case sensitivity issues in namespace names

    Returns:
        List of matches from the first successful namespace query
    """
    # Try primary namespace first
    logging.info(
        f"🔍 Querying primary namespace: '{primary_namespace}' in {idx_name}")
    matches = _query_index_with_batches(
        idx_name, primary_namespace, filter_dict, top_k)

    if matches:
        logging.info(
            f"✅ Found {len(matches)} matches in primary namespace: {primary_namespace}")
        return matches

    # No matches in primary - try fallback namespaces
    logging.warning(
        f"⚠️ No matches in primary namespace '{primary_namespace}', trying fallbacks...")

    # Define fallback namespaces to try
    fallback_namespaces = [
        "maintenance_requests",
        "maintenance_jobs",
        "Maintenance_Requests",
        "Maintenance_Jobs",
        "planon_data",
        None,  # Try default namespace
    ]

    # Remove the primary namespace from fallbacks to avoid duplicate query
    fallback_namespaces = [
        ns for ns in fallback_namespaces if ns != primary_namespace]

    for fallback_ns in fallback_namespaces:

        display_ns = get_display_namespace(fallback_ns)
        try:

            logging.info(f"  🔄 Trying fallback namespace: {display_ns}")

            matches = _query_index_with_batches(
                idx_name, fallback_ns, filter_dict, top_k)

            if matches:
                logging.warning(
                    f"✅ SUCCESS: Found {len(matches)} matches in fallback namespace: {display_ns}\n"
                    f"   ⚠️ CONFIGURATION ISSUE: Update NAMESPACE_MAPPINGS to use '{fallback_ns}' instead of '{primary_namespace}'"
                )
                return matches

        except Exception as e:
            logging.debug(
                f"  Failed to query fallback namespace {display_ns}: {e}")
            continue

    logging.error(
        f"❌ No matches found in primary or any fallback namespaces for {idx_name}")
    return []


def diagnose_maintenance_namespaces(index_name: str = "local-docs") -> Dict[str, Any]:
    """
    Diagnostic function to identify where maintenance data actually exists in Pinecone.

    Reports:
    - Available namespaces
    - Document types found per namespace
    - Namespaces containing maintenance data
    """

    result = {
        "index": index_name,
        "namespaces": {},
        "maintenance_namespaces": [],
        "recommendations": []
    }

    try:
        idx = open_index(index_name)
        stats = idx.describe_index_stats()

        logging.info(f"\n{'='*60}")
        logging.info(f"📊 DIAGNOSTIC: Index stats for '{index_name}'")
        logging.info(f"{'='*60}")
        logging.info(
            f"  Total vectors: {stats.get('total_vector_count', 0):,}")
        logging.info(f"  Dimension: {stats.get('dimension', 0)}")

        namespaces = stats.get("namespaces", {})

        if not namespaces:
            logging.warning("⚠️ No namespaces found in index!")
            result["recommendations"].append(
                "CRITICAL: Index has no namespaces — check ingestion."
            )
            return result

        # ----------------------------------------------------
        # Inspect each namespace
        # ----------------------------------------------------
        for ns_name, ns_stats in namespaces.items():
            vector_count = ns_stats.get("vector_count", 0)

            result["namespaces"][ns_name] = {
                "vector_count": vector_count,
                "doc_types": set()
            }

            display_ns = ns_name or "__default__"
            logging.info(
                f"\n  Namespace '{display_ns}': {vector_count:,} vectors")

            if vector_count == 0:
                continue

            try:
                dim = stats.get("dimension", 1536)
                zero_vec = [0.0] * dim

                # Query Pinecone for a sample
                response = idx.query(
                    vector=zero_vec,
                    top_k=min(100, vector_count),
                    namespace=ns_name or None,
                    include_metadata=True
                )

                # ✅ Convert to dictionary — safe for Pylance
                # ✅ Tell Pylance to treat the response as Any (runtime-safe)
                response_dict: Dict[str, Any] = cast(Any, response).to_dict()

                matches: List[Dict[str, Any]
                              ] = response_dict.get("matches", [])

                doc_types: Dict[str, int] = {}
                building_count = 0

                for match in matches:
                    md = match.get("metadata", {}) or {}
                    doc_type = md.get("document_type", "unknown")

                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                    if md.get("building_name") or md.get("canonical_building_name"):
                        building_count += 1

                result["namespaces"][ns_name]["doc_types"] = set(
                    doc_types.keys())

                # Logging info
                logging.info(f"    Document types found:")
                for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
                    logging.info(f"      - {dt}: {count} vectors (sample)")

                    if "maintenance" in dt.lower():
                        result["maintenance_namespaces"].append({
                            "namespace": ns_name,
                            "doc_type": dt,
                            "sample_count": count
                        })

                logging.info(
                    f"    Vectors with building metadata: {building_count}/{len(matches)}"
                )

            except Exception as e:
                logging.warning(
                    f"⚠️ Could not sample namespace '{display_ns}': {e}")
                result["namespaces"][ns_name]["error"] = str(e)

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------
        logging.info(f"\n{'='*60}")
        logging.info("📋 RECOMMENDATIONS")
        logging.info(f"{'='*60}")

        if result["maintenance_namespaces"]:
            logging.info(
                f"✅ Found maintenance data in "
                f"{len(result['maintenance_namespaces'])} namespace(s):"
            )

            for item in result["maintenance_namespaces"]:
                ns = item["namespace"] or "__default__"
                logging.info(
                    f"   - Namespace: {ns}\n"
                    f"     Doc type: {item['doc_type']}\n"
                    f"     Sample count: {item['sample_count']}"
                )
                result["recommendations"].append(
                    f"✅ Use namespace '{ns}' for doc_type '{item['doc_type']}'"
                )
        else:
            logging.warning("⚠️ No maintenance data found in any namespace")
            result["recommendations"].extend([
                "❌ No maintenance data detected",
                "💡 Verify maintenance data was ingested",
                "💡 Ensure `document_type` metadata is being set",
                "💡 Validate namespace mapping during ingestion"
            ])

        logging.info(f"{'='*60}\n")

        return result

    except Exception as e:
        logging.error(f"❌ Diagnostic failed: {e}", exc_info=True)
        result["error"] = str(e)
        result["recommendations"].append(f"CRITICAL ERROR: {e}")
        return result


def create_maintenance_building_filter(building_name: str) -> Dict[str, Any]:
    """
    Create a Pinecone filter for a building name, checking both
    canonical_building_name and building_name fields.
    """
    norm_building = normalise_building_name(building_name)
    return {
        "$or": [
            {"canonical_building_name": {"$eq": norm_building}},
            {"building_name": {"$eq": norm_building}},
        ]
    }


def create_document_building_filter(building_name: str) -> Dict[str, Any]:
    """
    Create a Pinecone filter for a building name, checking only
    canonical_building_name.
    """
    norm_building = normalise_building_name(building_name)
    return {
        "$or": [
            {"canonical_building_name": {"$eq": norm_building}}
        ]
    }


def flatten_request_metrics(metrics: dict) -> dict[str, dict[str, int]]:
    """
    Convert:
       category -> priority -> status -> count
    into:
       category -> status -> count
    Summing across priorities.
    """
    flat = {}
    for cat, priorities in metrics.items():
        if not isinstance(priorities, dict):
            continue
        for priority, statuses in priorities.items():
            if not isinstance(statuses, dict):
                continue
            for status, count in statuses.items():
                if not isinstance(count, int):
                    continue
                flat.setdefault(cat, {})
                flat[cat][status] = flat[cat].get(status, 0) + count
    return flat


def _filter_maintenance_buildings(
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
      • priority (P1-P6, PPM, Other)
      • status (open, complete, in progress, etc.)

    Supports:
      • 3-layer request metrics: category -> priority -> status -> count
      • 2-layer job metrics:     category -> status -> count
    """

    category = category.lower() if category else None
    status = status.lower() if status else None
    priority_norm = normalise_priority(
        priority).strip().lower() if priority else None

    filtered: list[dict] = []

    # -------------------------------------------------------------
    # Iterate through all returned building vectors
    # -------------------------------------------------------------
    for m in matches:
        md = m.get("metadata", {}) or {}
        raw_metrics = md.get("maintenance_metrics", {})

        # ---------------------------------------------------------
        # Parse / normalise metrics safely
        # ---------------------------------------------------------
        metrics: dict = {}

        # Metrics sometimes stored as JSON strings
        if isinstance(raw_metrics, str):
            try:
                metrics = json.loads(raw_metrics)
            except Exception:
                logging.warning(
                    f"⚠️ Invalid maintenance_metrics for building "
                    f"{md.get('canonical_building_name') or md.get('building_name')}"
                )
                metrics = {}
        elif isinstance(raw_metrics, dict):
            metrics = raw_metrics
        else:
            metrics = {}

        # ---------------------------------------------------------
        # Detect whether this is a 3-layer structure (requests)
        #    category → priority → status
        # or a 2-layer structure (jobs)
        #    category → status
        # ---------------------------------------------------------
        has_priority = any(
            isinstance(layer, dict)
            and any(isinstance(sub, dict) for sub in layer.values())
            for layer in metrics.values()
        )

        # ---------------------------------------------------------
        # Normalise to lowercase and ensure all counts are ints
        # ---------------------------------------------------------
        if has_priority:
            # Requests
            metrics_norm = {
                cat.lower(): {
                    prio.lower(): {
                        stat.lower(): c
                        for stat, c in status_map.items()
                        if isinstance(c, int)
                    }
                    for prio, status_map in prio_map.items()
                    if isinstance(status_map, dict)
                }
                for cat, prio_map in metrics.items()
                if isinstance(prio_map, dict)
            }
        else:
            # Jobs
            metrics_norm = {
                cat.lower(): {
                    stat.lower(): c
                    for stat, c in status_map.items()
                    if isinstance(c, int)
                }
                for cat, status_map in metrics.items()
                if isinstance(status_map, dict)
            }

        # ---------------------------------------------------------
        # Building filter
        # ---------------------------------------------------------
        bname = (
            md.get("canonical_building_name")
            or md.get("building_name")
            or ""
        )

        if building and building.lower() not in bname.lower():
            continue

        # ---------------------------------------------------------
        # Category filter
        # ---------------------------------------------------------
        if category and category not in metrics_norm:
            continue

        # ---------------------------------------------------------
        # Priority filter (only applies to requests)
        # ---------------------------------------------------------
        if priority_norm and has_priority:
            if category:
                if priority_norm not in metrics_norm.get(category, {}):
                    continue
            else:
                # Search across all categories
                if not any(
                    priority_norm in prio_map
                    for prio_map in metrics_norm.values()
                ):
                    continue

        # ---------------------------------------------------------
        # Status filter — fully corrected logic
        # ---------------------------------------------------------
        if status:
            count_int = 0

            # -----------------------------------------------------
            # Category specified
            # -----------------------------------------------------
            if category:
                cat_data = metrics_norm.get(category, {})

                if has_priority:
                    # Requests: sum contributions from all priorities
                    vals: list[int] = []
                    if isinstance(cat_data, dict):
                        for prio_data in cat_data.values():
                            if isinstance(prio_data, dict):
                                vals.append(int(prio_data.get(status, 0)))
                    count_int = sum(vals)

                else:
                    # Jobs: cat → status
                    if isinstance(cat_data, dict):
                        val = cat_data.get(status, 0)
                        if isinstance(val, int):
                            count_int = val
                        elif isinstance(val, dict):
                            # Rare but safe: sum nested dict of ints
                            count_int = sum(
                                v for v in val.values() if isinstance(v, int))

            # -----------------------------------------------------
            # No category specified → search all categories
            # -----------------------------------------------------
            else:
                if has_priority:
                    # Flatten priority layer across all categories
                    flat = flatten_request_metrics(metrics_norm)
                    vals: list[int] = []
                    for stats in flat.values():
                        if isinstance(stats, dict):
                            vals.append(int(stats.get(status, 0)))
                    count_int = sum(vals)

                else:
                    # Jobs: categories → statuses
                    vals: list[int] = []
                    for cat_map in metrics_norm.values():
                        if isinstance(cat_map, dict):
                            val = cat_map.get(status, 0)
                            if isinstance(val, int):
                                vals.append(val)
                            elif isinstance(val, dict):
                                vals.append(
                                    sum(v for v in val.values() if isinstance(v, int)))
                    count_int = sum(vals)

            # Skip buildings with zero total for this status
            if count_int <= 0:
                continue

        # ---------------------------------------------------------
        # Passed all filters
        # ---------------------------------------------------------
        filtered.append(m)

    return filtered

# -----------------------------------------------------------------------------
# Maintenance Query Logic
# -----------------------------------------------------------------------------


def generate_maintenance_answer(
    query: str,
    building_override: str | None = None,
) -> Optional[str]:
    """
    Handles maintenance queries using building-level vectors in Pinecone.
    Filters on metadata, then delegates formatting to maintenance_utils.
    """
    logging.info(f"🔍 MAINTENANCE QUERY: '{query}'")
    from maintenance_utils import (
        format_multi_building_metrics,   # <-- use the multi-building formatter
    )

    # Ensure cache is ready
    BuildingCacheManager.ensure_initialised()

    # --- Parse query ---
    if not BuildingCacheManager.is_populated():
        logging.warning(
            "⚠️ Building cache not populated — no building filtering possible")
        known_buildings = []
    else:
        known_buildings = BuildingCacheManager.get_known_buildings()

    parsed = parse_maintenance_query(query, known_buildings=known_buildings)
    logging.info(f"📋 PARSED: {parsed}")

    building = parsed.get("building_name")
    category = parsed.get("category")
    priority = parsed.get("priority")
    status = parsed.get("status")
    query_type = parsed.get("query_type") or "requests"

    # 🔁 NEW: allow context-provided building to override missing one
    if (not building) and building_override:
        logging.info(
            f"🧠 Using building from context override: {building_override}"
        )
        building = building_override

    logging.info(f"\n🔧 MAINTENANCE QUERY ANALYSIS")
    logging.info(f"  Query: {query}")
    logging.info(f"  Building: {building}")
    logging.info(f"  Category: {category}")
    logging.info(f"  Status: {status}")
    logging.info(f"  Query type: {query_type}")

    # Ignore mis-detection like "maintenance"
    if building and building.lower() in INVALID_BUILDING_NAMES:
        logging.info(f"⚠️ Ignoring mis-detected building name: {building}")
        building = None

    # --- Namespace selection ---
    namespace = "maintenance_jobs" if query_type == "jobs" else "maintenance_requests"
    logging.info(f"🔎 Using Pinecone namespace: {namespace}")

    idx = open_index("local-docs")
    ns_details = idx.describe_index_stats().get("namespaces", {})
    if namespace not in ns_details:
        logging.warning(
            f"⚠️ Namespace '{namespace}' not found — available: {list(ns_details.keys())}")

    # --- Query Pinecone for all vectors in namespace ---
    dim = idx.describe_index_stats().get("dimension", 1536)
    zero_vec = [0.0] * dim

    raw: Any = idx.query(
        vector=zero_vec,
        top_k=2000,
        namespace=namespace,
        include_metadata=True,
    )
    response = raw.to_dict() if hasattr(
        raw, "to_dict") else cast(Dict[str, Any], raw)
    matches = response.get("matches", [])
    logging.info(f"Retrieved {len(matches)} maintenance building vectors")

    # --- Filter by building/category/status ---
    filtered = _filter_maintenance_buildings(
        matches, building, category, priority, status)
    logging.info(f"Filtered matches: {len(filtered)}")
    if not filtered:
        return "No buildings match that maintenance query."

    # --- Build a deduped map: building -> status_totals ---
    building_status_map: Dict[str, Dict[str, int]] = {}

    for m in filtered:
        md = m.get("metadata", {}) or {}
        bname = md.get("canonical_building_name") or md.get(
            "building_name") or "Unknown building"

        metrics = md.get("maintenance_metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        if not isinstance(metrics, dict):
            continue

        # Detect 3-level (requests) vs 2-level (jobs)
        has_priority = any(isinstance(v, dict) and
                           any(isinstance(subv, dict) for subv in v.values())
                           for v in metrics.values())

        if has_priority:
            collapsed = flatten_request_metrics(metrics)
        else:
            collapsed = metrics  # already: category -> status -> count

        status_totals = {}
        for cat, statuses in collapsed.items():
            for s, c in statuses.items():
                if isinstance(c, int):
                    s_norm = s.lower()
                    status_totals[s_norm] = status_totals.get(s_norm, 0) + c

        # merge by taking max per status to avoid double-counting same totals
        if bname not in building_status_map:
            building_status_map[bname] = status_totals
        else:
            for k, v in status_totals.items():
                building_status_map[bname][k] = max(
                    building_status_map[bname].get(k, 0), v)

    # --- Compute totals BEFORE slicing ---
    ranked = sorted(
        ((b, sum(stats.values())) for b, stats in building_status_map.items()),
        key=lambda x: -x[1],
    )
    total_buildings = len(ranked)
    total_records = sum(total for _, total in ranked)

    # --- Take top 10 buildings for display ---
    top_buildings = [b for b, _ in ranked[:10]]
    trimmed_stats: Dict[str, Dict[str, int]] = {
        b: building_status_map[b] for b in top_buildings}

    # ✅ Delegate the entire rendering to the multi-building formatter
    return format_multi_building_metrics(
        building_stats=trimmed_stats,
        total_buildings=total_buildings,
        total_records=total_records,
        query_type=query_type,
        limit=10,
    )

# -----------------------------------------------------------------------------
# Property Condition + Ranking
# -----------------------------------------------------------------------------


def generate_property_condition_answer(query: str) -> Optional[str]:
    condition = normalise_property_condition(query)
    if not condition:
        return "Please specify a property condition (e.g., Condition A, Condition B, Derelict)."

    results = query_property_by_condition(condition)

    building_count = results['building_count']
    if building_count == 0:
        return f"No buildings found with property condition '{condition}'."

    answer = f"**{building_count} {_plural(building_count, 'building')}** are **{condition}**\n\n"

    for i, building in enumerate(results['matching_buildings'][:5], 1):
        answer += f"{i}. **{building['building_name']}**\n"
        answer += f"   - Condition: {building['condition']}\n"
        if building.get('campus'):
            answer += f"   - Campus: {building['campus']}\n"
        if building.get('gross_area'):
            answer += f"   - Gross Area: {building['gross_area']} sq m\n"
        answer += "\n"

    if building_count > 5:
        answer += f"➕ ... and {building_count - 5} more buildings."

    return answer


def generate_ranking_answer(query: str) -> Optional[str]:
    """
    Generate an answer for building ranking queries (by area).
    Works with rank_buildings_by_area() structured output.
    """
    logging.info("🔢 generate_ranking_answer() called with query: '%s'", query)
    q = query.lower()

    # Determine area type
    area_type = 'gross' if 'net' not in q else 'net'

    # Determine order
    order = 'desc'
    if 'smallest' in q or 'ascending' in q:
        order = 'asc'

    # Determine limit (Top N etc.)
    limit: Optional[int] = None
    top_match = re.search(r'top\s+(\d+)', q)
    if top_match:
        limit = int(top_match.group(1))
    elif not re.search(r"\ball\s+buildings?\b", q):
        limit = 10  # default limit to avoid dumping all buildings
    logging.info("   Parsed parameters - area_type='%s', order='%s', limit=%s",
                 area_type, order, limit)

    # Run structured ranking
    results = rank_buildings_by_area(
        area_type=area_type, order=order, limit=limit)

    logging.info("   rank_buildings_by_area() returned results")

    total_buildings = results.get('total_buildings', 0)
    ranked = results.get('results', [])

    logging.info("   Results: total_buildings=%d, ranked_results=%d",
                 total_buildings, len(ranked))

    if total_buildings == 0 or not ranked:
        return "No buildings found with area data."

    # Pretty labels
    order_text = "largest" if order == 'desc' else "smallest"
    area_text = "gross" if area_type == 'gross' else "net"

    logging.info(
        "   Generating formatted answer for %d buildings", len(ranked))

    # Build answer
    answer = f"**Buildings ranked by {area_text} area ({order_text} first):**\n\n"
    answer += f"**Total buildings with area data:** {total_buildings}\n\n"

    # Render each ranked record
    for item in ranked:
        rank = item.get('rank')
        name = item.get('building_name', 'Unknown')
        area_val = item.get('value')
        meta = item.get('metadata', {}) or {}

        answer += f"**{rank}. {name}**\n"
        if area_val is not None:
            answer += f"   - {area_text.title()} Area: {area_val:,.0f} sq m\n"

        # Optional metadata fields if present
        campus = meta.get('campus')
        if campus:
            answer += f"   - Campus: {campus}\n"

        condition = meta.get('condition')
        if condition:
            answer += f"   - Condition: {condition}\n"

        answer += "\n"
    logging.info("✅ generate_ranking_answer() completed - returning formatted answer (%d chars)",
                 len(answer))
    return answer

# -----------------------------------------------------------------------------
# Counting
# -----------------------------------------------------------------------------


def generate_counting_answer(query: str) -> Optional[str]:
    # Route to other handlers first
    if is_maintenance_query(query):
        return generate_maintenance_answer(query)
    if is_property_condition_query(query):
        return generate_property_condition_answer(query)
    if is_ranking_query(query):
        return generate_ranking_answer(query)
    if not is_counting_query(query):
        return None

    # Parse building + doc type
    known = BuildingCacheManager.get_known_buildings(
    ) if BuildingCacheManager.is_populated() else []
    building = extract_building_from_query(query, known)
    if building and building.lower() in INVALID_BUILDING_NAMES:
        building = None

    doc_type = extract_document_type_from_query(query)

    # Pinecone filters
    filter_dict = None
    if building:
        if doc_type in ["fire_risk_assessment", "operational_doc"]:
            filter_dict = create_document_building_filter(
                building)  # added placeholder
        else:
            filter_dict = create_maintenance_building_filter(building)

    if doc_type:
        doc_filter = {"document_type": {"$eq": doc_type}}
        filter_dict = {"$and": [filter_dict, doc_filter]
                       } if filter_dict else doc_filter

    # Query and aggregate
    building_set = set()
    keys_by_bldg = defaultdict(set)

    for idx in TARGET_INDEXES:

        ns = resolve_namespace(doc_type)  # based on query intent
        matches = _query_index_with_batches(idx, ns, filter_dict)
        for m in matches:
            md = m.get("metadata", {}) or {}
            b = md.get("canonical_building_name") or md.get(
                "UsrFRACondensedPropertyName")
            key = md.get("key")
            if not b:
                aliases = md.get("building_aliases")
                if isinstance(aliases, list) and aliases:
                    b = aliases[0]
            if not b:
                continue  # Skip entries with no building name
            b_norm = normalise_building_name(b)
            building_set.add(b_norm)
            # docs_by_bldg[b_norm] += 1
            if key:
                keys_by_bldg[b_norm].add(key)

    buildings = sorted(building_set)
    count = len(building_set)

    # --- Build Response
    if doc_type:
        doc_name = DOC_TYPE_NAMES_SIMPLE.get(doc_type, doc_type)
        if count == 0:
            return f"No buildings found with **{doc_name}** records."

        ans = f"**{count} buildings** have **{doc_name}**.\n\n"
        ans += f"**Total documents:** {sum(len(keys) for keys in keys_by_bldg.values())}\n\n"

        show = min(10, count)
        ans += f"**Listing {show} building(s):**\n"
        for b in buildings[:show]:
            docs_b = len(keys_by_bldg[b])
            ans += f"- **{b}** ({docs_b} {_plural(docs_b, 'doc')})\n"
            example_keys = list(keys_by_bldg[b])[:3]
            for key in example_keys:
                ans += f"  - `{key}`\n"
            remaining = len(keys_by_bldg[b]) - len(example_keys)
            if remaining > 0:
                ans += f"  - ... and {remaining} more\n"
        # Add "x additional buildings not shown"
        hidden = count - show
        if hidden > 0:
            ans += f"\n➕ …and **{hidden} additional {_plural(hidden, 'building')}** not shown."
        return ans

    # No doc type → count total buildings
    if building:
        return f"**{building}** appears in **{count} buildings in the system.**"

    return f"**{count} unique buildings** are indexed in the system."
