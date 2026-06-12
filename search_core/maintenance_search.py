# search_core/maintenance_search.py

from search_core.generate_maintenance_answers import generate_maintenance_answer


def maintenance_search(instruction):
    answer = generate_maintenance_answer(
        instruction.query,
        access_filter=getattr(instruction, "access_filter", None),
    )
    return [], answer
