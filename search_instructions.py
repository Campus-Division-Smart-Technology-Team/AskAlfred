from dataclasses import dataclass


@dataclass
class SearchInstructions:
    type: str             # e.g. "semantic", "planon", "maintenance"
    query: str
    top_k: int
    building: str | None = None
    document_type: str | None = None
