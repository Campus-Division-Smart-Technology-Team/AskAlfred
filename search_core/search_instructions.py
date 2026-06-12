from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchInstructions:
    type: str  # e.g. "semantic", "planon", "maintenance"
    query: str
    top_k: int
    building: str | None = None
    document_type: str | None = None
    access_filter: dict[str, Any] = field(default_factory=dict)
