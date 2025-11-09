# search_core/maintenance_search.py

from structured_queries import generate_maintenance_answer


def maintenance_search(instruction):
    answer = generate_maintenance_answer(instruction.query)
    return [], answer
