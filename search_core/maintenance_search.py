# search_core/maintenance_search.py

from generate_maintenance_answers import generate_maintenance_answer


def maintenance_search(instruction):
    answer = generate_maintenance_answer(instruction.query)
    return [], answer
