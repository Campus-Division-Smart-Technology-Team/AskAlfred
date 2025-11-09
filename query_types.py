# query_types.py
from enum import Enum

# ============================================================================
# ENUMS
# ============================================================================


class QueryType(Enum):
    """All supported query types."""
    CONVERSATIONAL = "conversational"
    COUNTING = "counting"
    MAINTENANCE = "maintenance"
    RANKING = "ranking"
    PROPERTY_CONDITION = "property_condition"
    SEMANTIC_SEARCH = "semantic_search"
    ERROR = "error"
