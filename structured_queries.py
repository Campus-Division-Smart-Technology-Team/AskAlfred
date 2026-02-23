#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counting, ranking and maintenance query routing for AskAlfred
Full production logic with Pinecone access
Uses maintenance_utils for parsing and formatting
Logging at INFO level for query tracing
"""

from __future__ import annotations
from typing import Any, cast, Optional
import logging
import re
from collections import defaultdict
from maintenance_utils import _plural
from pinecone_utils import open_index, query_all_chunks
from config import TARGET_INDEXES, _route_namespace, normalise_ns, get_display_namespace
from building import (BuildingCacheManager,
                      normalise_building_name,
                      extract_building_from_query)
from building import sanitise_building_candidate
from generate_maintenance_answers import generate_maintenance_answer
from emojis import (EMOJI_TICK, EMOJI_CHART, EMOJI_REPEAT, EMOJI_BRAIN,
                    EMOJI_CROSS, EMOJI_CLIPBOARD, EMOJI_CAUTION, EMOJI_SEARCH, EMOJI_INIT,)

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
# Initialise logging
# -----------------------------------------------------------------------------
logging.info("%s structured_queries.py (FULL FUNCTIONAL) loaded", EMOJI_TICK)

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
    "requests": "maintenance_request",
    "job": "maintenance_job",
    "jobs": "maintenance_job",
}

DOC_TYPE_NAMES_SIMPLE = {
    "fire_risk_assessment": "Fire Risk Assessments",
    "operational_doc": "BMS/Operational",
    "planon_data": "Planon property data",
    "maintenance_request": "Maintenance Requests",
    "maintenance_job": "Maintenance Jobs",
    "unknown": "Unknown document types",
}

DEFAULT_BATCH_TOP_K = 1000
MAX_SAFE_TOP_K = 10000

# -----------------------------------------------------------------------------
# Detection helpers
# -----------------------------------------------------------------------------


def is_counting_query(query: str) -> bool:
    q = query.lower().strip()
    return any(p.search(q) for p in COUNTING_PATTERNS)


def is_maintenance_query(query: str) -> bool:
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


def is_yes_no_property_condition_question(query: str) -> bool:
    q = (query or "").strip().lower()
    # Covers: "is X derelict", "is X in condition b", "does X have condition c", etc.
    return bool(re.search(r"^(is|are|does|do)\b", q))


def query_property_by_condition(condition: str,
                                building_filter: Optional[str] = None,
                                index_names: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Query buildings by property condition from the 'planon_data' namespace.
    Optionally filter by a specific building.
    """
    if index_names is None:
        index_names = TARGET_INDEXES

    namespace = 'planon_data'
    canonical_condition = condition
    matching_buildings: list[dict[str, Any]] = []

    building_l = building_filter.lower().strip() if building_filter else None

    # Create Pinecone filter
    filter_dict = None
    if building_filter:
        # Use the existing building filter creation function
        filter_dict = create_building_filter(building_filter)

    for idx_name in index_names:
        matches = _query_index_with_batches(
            idx_name=idx_name,
            namespace=namespace,
            filter_dict=filter_dict,
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
) -> dict[str, Any]:
    """
    Rank buildings by gross or net area from Planon data vectors in Pinecone.

    Args:
        area_type: "gross" or "net" (default: gross)
        order: "asc" or "desc" (default: desc)
        limit: max number of buildings to return (default: None → all)

    Returns:
        dict structured for generate_ranking_answer().
    """
    logging.info("%s rank_buildings_by_area() called - area_type='%s', order='%s', limit=%s", EMOJI_CHART,
                 area_type, order, limit)

    # Which metadata field to read
    area_key = (
        'Property gross area (sq m)' if area_type == 'gross'
        else 'Property net area (sq m)'
    )
    logging.info("   Using metadata field: '%s'", area_key)

    namespace = "planon_data"
    buildings: dict[str, dict[str, Any]] = {}

    # Fetch all vectors from each target index
    logging.info("%s Querying %d target indexes for namespace '%s'", EMOJI_INIT,
                 len(TARGET_INDEXES), namespace)
    for idx in TARGET_INDEXES:
        matches = _query_index_with_batches(
            idx_name=idx,
            namespace=namespace,
            filter_dict=None,
            top_k=DEFAULT_BATCH_TOP_K
        )
        logging.info("%s Index '%s': Retrieved %d matches",
                     EMOJI_TICK, idx, len(matches))

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
    logging.info("%s Total buildings with valid area data: %d",
                 EMOJI_CLIPBOARD, total)
    if total == 0:
        logging.warning(
            "%s No buildings found with area data - returning empty results", EMOJI_CAUTION)
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
    logging.info("%s Buildings sorted by area (%s order)", EMOJI_TICK,
                 "descending" if reverse else "ascending")

    # Apply limit if given
    if limit is not None:
        original_count = len(ranked)
        ranked = ranked[:limit]
        logging.info(
            "%s Applied limit: %d (reduced from %d buildings)", EMOJI_BRAIN, limit, original_count)
    else:
        logging.info(
            "%s No limit applied; returning all %d buildings", EMOJI_CAUTION, len(ranked))

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
        "%s rank_buildings_by_area() completed - returning %d ranked buildings", EMOJI_TICK, len(results))

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
    filter_dict: Optional[dict[str, Any]] = None,
    top_k: int = DEFAULT_BATCH_TOP_K,
) -> list[dict[str, Any]]:
    """
    Query a Pinecone index in batches with optional filters.
    Returns all matches from all batches combined and deduplicated.
    """
    namespace = normalise_ns(namespace)
    display_namespace = get_display_namespace(namespace)

    try:
        idx = open_index(idx_name)
        stats = idx.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        total_vecs = ns_stats.get("vector_count", 0)

        if total_vecs == 0:
            print(
                "DEBUG: namespace='%s' raw_stats_keys=%s", namespace, stats.get('namespaces', {}).keys())
            logging.info(
                "[%s] namespace=%s has 0 vectors", idx_name, namespace)
            return []

        safe_top_k = min(top_k, MAX_SAFE_TOP_K)
        logging.info(
            "[%s] Fetching up to %d vectors from namespace=%s", idx_name, safe_top_k, display_namespace)
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
            "[%s] Retrieved %d unique matches (from %d total)", idx_name, len(deduped), len(all_matches))

        return deduped

    except Exception as e:
        logging.error("Error querying %s: %s", idx_name, e)
        return []


def _query_index_with_fallback(
    idx_name: str,
    primary_namespace: Optional[str],
    filter_dict: Optional[dict[str, Any]] = None,
    top_k: int = DEFAULT_BATCH_TOP_K,
) -> list[dict[str, Any]]:
    """
    Query index with fallback to alternative namespaces if primary returns no results.

    This handles cases where:
    - Namespace mapping is incorrect
    - Data was ingested with different namespace than expected
    - Case sensitivity issues in namespace names

    Returns:
        list of matches from the first successful namespace query
    """
    # Try primary namespace first
    logging.info(
        "%s Querying primary namespace: '%s' in %s", EMOJI_SEARCH, primary_namespace, idx_name)
    matches = _query_index_with_batches(
        idx_name, primary_namespace, filter_dict, top_k)

    if matches:
        logging.info(
            "%s Found %s matches in primary namespace: %s", EMOJI_TICK, len(matches), primary_namespace)
        return matches

    # No matches in primary - try fallback namespaces
    logging.warning(
        "%s No matches in primary namespace '%s', trying fallbacks...", EMOJI_CAUTION, primary_namespace)

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

            logging.info("%s Trying fallback namespace: %s",
                         EMOJI_REPEAT, display_ns)

            matches = _query_index_with_batches(
                idx_name, fallback_ns, filter_dict, top_k)

            if matches:
                logging.warning(
                    "%s SUCCESS: Found %s matches in fallback namespace: %s\n"
                    " %s CONFIGURATION ISSUE: Update NAMESPACE_MAPPINGS to use '%s' instead of '%s'", EMOJI_TICK,
                    len(matches), display_ns, EMOJI_CAUTION, fallback_ns, primary_namespace
                )
                return matches

        except Exception as e:
            logging.debug(
                "%s Failed to query fallback namespace %s: %s", EMOJI_CAUTION, display_ns, e)
            continue

    logging.error(
        "%s No matches found in primary or any fallback namespaces for %s", EMOJI_CROSS, idx_name)
    return []


def diagnose_maintenance_namespaces(index_name: str = "local-docs") -> dict[str, Any]:
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

        logging.info("\n%s", "="*60)
        logging.info("%s DIAGNOSTIC: Index stats for '%s'",
                     EMOJI_CHART, index_name)
        logging.info("%s", "="*60)
        logging.info(
            "  Total vectors: %d", stats.get('total_vector_count', 0))
        logging.info("  Dimension: %d", stats.get('dimension', 0))

        namespaces = stats.get("namespaces", {})

        if not namespaces:
            logging.warning("%s No namespaces found in index!", EMOJI_CAUTION)
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
                "\n  Namespace '%s': %d vectors", display_ns, vector_count)

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
                response_dict: dict[str, Any] = cast(Any, response).to_dict()

                matches: list[dict[str, Any]
                              ] = response_dict.get("matches", [])

                doc_types: dict[str, int] = {}
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
                logging.info("    Document types found:")
                for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
                    logging.info("      - %s: %d vectors (sample)", dt, count)

                    if "maintenance" in dt.lower():
                        result["maintenance_namespaces"].append({
                            "namespace": ns_name,
                            "doc_type": dt,
                            "sample_count": count
                        })

                logging.info(
                    "    Vectors with building metadata: %d/%d", building_count, len(
                        matches)
                )

            except Exception as e:
                logging.warning(
                    "%s Could not sample namespace '%s': %s", EMOJI_CAUTION, display_ns, e)
                result["namespaces"][ns_name]["error"] = str(e)

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------
        logging.info("\n%s", "="*60)
        logging.info("%s RECOMMENDATIONS", EMOJI_CLIPBOARD)
        logging.info("%s", "="*60)

        if result["maintenance_namespaces"]:
            logging.info(
                "%s Found maintenance data in %d namespace(s):", EMOJI_TICK,
                len(result['maintenance_namespaces'])
            )

            for item in result["maintenance_namespaces"]:
                ns = item["namespace"] or "__default__"
                logging.info(
                    "   - Namespace: %s\n     Doc type: %s\n     Sample count: %d",
                    ns, item['doc_type'], item['sample_count']
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

        logging.info("%s\n", "="*60)

        return result

    except Exception as e:
        logging.error("%s Diagnostic failed: %s",
                      EMOJI_CROSS, e, exc_info=True)
        result["error"] = str(e)
        result["recommendations"].append(f"CRITICAL ERROR: {e}")
        return result


def create_building_filter(building_name: str) -> dict[str, Any]:
    """
    Create a Pinecone filter for a building name, checking both
    canonical_building_name and building_name fields.
    """
    raw = (building_name or "").strip()
    norm = normalise_building_name(raw)

    # De-duplicate if normalise_building_name doesn't change it
    candidates = [raw] if raw == norm else [raw, norm]

    return {
        "$or": [
            {"canonical_building_name": {"$in": candidates}},
            {"building_name": {"$in": candidates}},
            # Optional: if you ingest aliases as a list, this helps a lot
            {"building_aliases": {"$in": candidates}},
            # Optional: if Planon name is stored under this field in metadata
            {"UsrFRACondensedPropertyName": {"$in": candidates}},
        ]
    }


def create_document_building_filter(building_name: str) -> dict[str, Any]:
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

# -----------------------------------------------------------------------------
# Property Condition + Ranking
# -----------------------------------------------------------------------------


def generate_property_condition_answer(query: str) -> Optional[str]:
    # condition = normalise_property_condition(query)
    # if not condition:
    #     return "Please specify a property condition (e.g., Condition A, Condition B, Derelict)."

    q = (query or "").strip()
    ql = q.lower()

    # Extract building name from query
    known = BuildingCacheManager.get_known_buildings(
    ) if BuildingCacheManager.is_populated() else []
    building = sanitise_building_candidate(
        extract_building_from_query(query, known)
    )
    condition = normalise_property_condition(query)
    yes_no_intent = is_yes_no_property_condition_question(query)

    # ------------------------------------------------------------------
    # CASE A: Building specified but NO condition specified
    # e.g. "What is the property condition of BDFI?"
    # ------------------------------------------------------------------
    if building and not condition:
        # Query Planon vectors for just this building, then read Property condition
        matches_found = []
        for idx_name in TARGET_INDEXES:
            matches = _query_index_with_batches(
                idx_name=idx_name,
                namespace="planon_data",
                filter_dict=create_building_filter(building),
                top_k=DEFAULT_BATCH_TOP_K,
            )
            matches_found.extend(matches)

        # Extract first non-empty property condition (dedupe-ish)
        prop_condition = None
        for m in matches_found:
            md = (m.get("metadata", {}) or {})
            val = md.get("Property condition")
            if isinstance(val, str) and val.strip():
                prop_condition = val.strip()
                break

        if not prop_condition:
            return f"I couldn't find a property condition for **{building}** in Planon data."

        return f"The property condition of **{building}** is **{prop_condition}**."
    # ------------------------------------------------------------------
    # CASE B: No building and no condition → prompt user (unchanged behavior)
    # ------------------------------------------------------------------
    if not building and not condition:
        return "Please specify a property condition (e.g., Condition A, Condition B, Derelict) or a building."

    # ------------------------------------------------------------------
    # CASE C: Condition specified (with or without building)
    # e.g. "Is BDFI derelict?" or "Which buildings are derelict?"
    # ------------------------------------------------------------------
    if condition:
        results = query_property_by_condition(
            condition, building_filter=building)
        building_count = results["building_count"]

        # If user asked about a specific building + yes/no intent → return boolean answer
        if building and yes_no_intent:
            if building_count > 0:
                return f"Yes, **{building}** is **{condition}**."
            return f"No, **{building}** is not **{condition}**."

        # If user asked about a specific building (not yes/no) → return direct statement, no list
        if building:
            if building_count == 0:
                return f"Building **{building}** does not have property condition **{condition}**."
            return f"**{building}** is **{condition}**."

        # Otherwise (no building) → listing/count output as before
        if building_count == 0:
            return f"No buildings found with property condition '{condition}'."

        answer = f"**{building_count} {_plural(building_count, 'building')}** are **{condition}**\n\n"
        for i, b in enumerate(results["matching_buildings"][:5], 1):
            answer += f"{i}. **{b['building_name']}**\n"
            answer += f"   - Condition: {b['condition']}\n"
            if b.get("campus"):
                answer += f"   - Campus: {b['campus']}\n"
            if b.get("gross_area"):
                answer += f"   - Gross Area: {b['gross_area']} sq m\n"
            answer += "\n"

        if building_count > 5:
            answer += f"➕ ... and {building_count - 5} more buildings."

        return answer

    # Fallback (should be unreachable)
    return None

    # results = query_property_by_condition(condition,  building_filter=building)

    # building_count = results['building_count']
    # if building_count == 0:
    #     # return f"No buildings found with property condition '{condition}'."
    #     if building:
    #         return f"Building '{building}' does not have property condition '{condition}'."
    #     return f"No buildings found with property condition '{condition}'."

    # if building:
    #     answer = f"**{building}** is **{condition}**\n\n"
    # else:
    #     answer = f"**{building_count} {_plural(building_count, 'building')}** are **{condition}**\n\n"

    # for i, building in enumerate(results['matching_buildings'][:5], 1):
    #     answer += f"{i}. **{building['building_name']}**\n"
    #     answer += f"   - Condition: {building['condition']}\n"
    #     if building.get('campus'):
    #         answer += f"   - Campus: {building['campus']}\n"
    #     if building.get('gross_area'):
    #         answer += f"   - Gross Area: {building['gross_area']} sq m\n"
    #     answer += "\n"

    # if building_count > 5:
    #     answer += f"➕ ... and {building_count - 5} more buildings."

    # return answer


def generate_ranking_answer(query: str) -> Optional[str]:
    """
    Generate an answer for building ranking queries (by area).
    Works with rank_buildings_by_area() structured output.
    """
    logging.info(
        "%s generate_ranking_answer() called with query: '%s'", EMOJI_INIT, query)
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

    total_buildings = results.get('total_buildings', 0)
    ranked = results.get('results', [])

    logging.info("%s Results: total_buildings=%d, ranked_results=%d", EMOJI_CHART,
                 total_buildings, len(ranked))

    if total_buildings == 0 or not ranked:
        return "No buildings found with area data."

    # Pretty labels
    order_text = "largest" if order == 'desc' else "smallest"
    area_text = "gross" if area_type == 'gross' else "net"

    logging.info(
        "%s Generating formatted answer for %d buildings", EMOJI_BRAIN, len(ranked))

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
    logging.info("%s generate_ranking_answer() completed - returning formatted answer (%d chars)", EMOJI_TICK,
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
    building = sanitise_building_candidate(
        extract_building_from_query(query, known)
    )

    doc_type = extract_document_type_from_query(query)

    # Pinecone filters
    filter_dict = None
    if building:
        if doc_type in ["fire_risk_assessment", "operational_doc"]:
            filter_dict = create_document_building_filter(
                building)  # added placeholder
        else:
            filter_dict = create_building_filter(building)

    if doc_type:
        doc_filter = {"document_type": {"$eq": doc_type}}
        filter_dict = {"$and": [filter_dict, doc_filter]
                       } if filter_dict else doc_filter

    # Query and aggregate
    building_set = set()
    keys_by_bldg = defaultdict(set)

    for idx in TARGET_INDEXES:

        ns = _route_namespace(doc_type)  # based on query intent
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
        ans += f"**listing {show} building(s):**\n"
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
