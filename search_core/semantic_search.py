# search_core/semantic_search.py

import logging
from typing import Optional

from building.utils import (
    extract_building_from_query,
    group_results_by_building,
    resolve_building_name_fuzzy,
)
from config import MIN_SCORE_THRESHOLD, TARGET_INDEXES
from domain.business_terms import BusinessTermMapper
from search_core.generate_semantic_answer import (
    enhanced_answer_with_source_date,
    generate_building_focused_answer,
)
from search_core.search_utils import (
    apply_building_boost,
    apply_doc_type_boost,
    apply_occupancy_capacity_boost,
    deduplicate_results,
    get_effective_score,
    search_one_index,
)


def semantic_search(
    query: str,
    top_k: int,
    building_filter: Optional[str] = None,
    access_filter: Optional[dict] = None,
) -> tuple[list[dict], str, str, bool]:

    logging.info(
        "[semantic_search] running: q=%s, k=%s, building=%s",
        query,
        top_k,
        building_filter,
    )

    # Extract building (or use preset)
    raw_building = building_filter or extract_building_from_query(query)
    building = resolve_building_name_fuzzy(raw_building)

    # Enhance business terms (FRA -> fire risk assessment)
    enhanced_query, term_context = BusinessTermMapper.enhance_query_with_terms(query)

    # Optional document-type boost
    doc_type_filter = None
    if term_context:
        first_term = list(term_context.values())[0]
        doc_type_filter = first_term.get("document_type")

    # ===== Stage 1 — building-filtered search =====
    results = []
    used_filter = False

    if building:
        for idx in TARGET_INDEXES:
            hits = search_one_index(
                idx,
                enhanced_query,
                top_k * 3,
                building_filter=building,
                access_filter=access_filter,
            )
            results.extend(hits)

        if results:
            used_filter = True
            results = deduplicate_results(results)

    # ===== Stage 2 — pure semantic fallback =====
    if not results:
        for idx in TARGET_INDEXES:
            hits = search_one_index(
                idx,
                enhanced_query,
                top_k * 3,
                building_filter=None,
                access_filter=access_filter,
            )
            results.extend(hits)

        results = deduplicate_results(results)

    # Apply doc type boost
    if doc_type_filter:
        results = apply_doc_type_boost(results, doc_type_filter)

    # Apply building boost (especially important in stage 2)
    if building:
        results = apply_building_boost(
            results, building, boost_factor=3.0 if not used_filter else 1.5
        )

    results = apply_occupancy_capacity_boost(results, enhanced_query)

    # Sort by boosted or base score
    results.sort(key=get_effective_score, reverse=True)
    top_hits = results[:top_k]

    # ===== Score threshold check =====
    score_too_low = False
    if top_hits:
        top_score = get_effective_score(top_hits[0])
        if top_score < MIN_SCORE_THRESHOLD:
            score_too_low = True
            return (
                top_hits,
                (
                    f"I found results, but the top match scored {top_score:.3f}, "
                    "which is below the threshold. Try rephrasing."
                ),
                "",
                True,
            )

    # ===== Answer generation =====
    if not top_hits:
        return [], "I couldn't find any matching documents.", "", True

    answer, pub_info = "", ""
    building_groups = group_results_by_building(top_hits)

    # Building-aware answer
    if building and building_groups.get(building):
        answer, pub_info, cited_results = generate_building_focused_answer(
            query, top_hits[0], top_hits, building, building_groups, term_context
        )
        # Reorder so [SN] citation tags resolve against the returned list:
        # cited sources first (in S-number order), remaining hits after.
        cited_ids = {id(result) for result in cited_results}
        top_hits = cited_results + [
            result for result in top_hits if id(result) not in cited_ids
        ]
    else:
        answer, pub_info = enhanced_answer_with_source_date(
            query, top_hits[0], top_hits, term_context, target_building=building
        )

    return top_hits, answer, pub_info, score_too_low
