# search_core/search_router.py

from typing import Any, Union, Optional
from search_instructions import SearchInstructions
from alfred_exceptions import RoutingError

from .semantic_search import semantic_search
from .planon_search import planon_search
from .maintenance_search import maintenance_search


# ------------------------------------------------------------------------------------
# Return type contracts (must match actual backend implementations)
# ------------------------------------------------------------------------------------

# semantic_search returns:
#   (results, answer, publication_info, score_too_low)
SemanticReturn = tuple[list[dict[str, Any]], str, str, bool]

# planon_search returns:
#   (results, answer, publication_info)
# The publication_info is always "" in the current implementation.
PlanonReturn = tuple[list[dict[str, Any]], Optional[str], str]

# maintenance_search returns:
#   (results, answer)
MaintenanceReturn = tuple[list[dict[str, Any]], Optional[str]]

# Unified router return type:
ReturnUnion = Union[SemanticReturn, PlanonReturn, MaintenanceReturn]


# ------------------------------------------------------------------------------------
# Router
# ------------------------------------------------------------------------------------


def execute(instr: SearchInstructions) -> ReturnUnion:
    """
    Unified router that delegates to the appropriate search backend based on
    the SearchInstructions.type field.

    Expected return shapes:
      - type == "semantic":    (List[Dict], str, str, bool)
      - type == "planon":      (List[Dict], str, str)
      - type == "maintenance": (List[Dict], str)
    """
    itype = getattr(instr, "type", None)

    # Semantic vector search
    if itype == "semantic":
        return semantic_search(
            query=instr.query,
            top_k=instr.top_k,
            building_filter=getattr(instr, "building", None),
        )

    # Planon structured search (property/condition/ranking)
    if itype == "planon":
        # planon_search returns: (results, answer, publication_info)
        return planon_search(instr)

    # Maintenance structured search
    if itype == "maintenance":
        # maintenance_search returns: (results, answer)
        return maintenance_search(instr)

    raise RoutingError(f"Unknown search instruction type: {itype}")
