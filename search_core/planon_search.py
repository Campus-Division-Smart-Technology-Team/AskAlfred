# search_core/planon_search.py
from structured_queries import (
    generate_property_condition_answer,
    generate_ranking_answer,
)


def planon_search(instruction):
    q = instruction.query.lower()

    # Ranking (e.g. "biggest buildings", "top 5 by area")
    if "rank" in q or "biggest" in q or "largest" in q:
        answer = generate_ranking_answer(instruction.query)
        return [], answer, ""

    # Property condition queries ("which buildings have asbestos?")
    answer = generate_property_condition_answer(instruction.query)
    return [], answer, ""
