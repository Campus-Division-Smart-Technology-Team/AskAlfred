# exceptions.py
class AskAlfredError(Exception):
    """Base exception for AskAlfred."""
    pass


class BuildingNotFoundError(AskAlfredError):
    """Raised when building cannot be found in cache."""
    pass


class HandlerError(AskAlfredError):
    """Raised when handler fails to process query."""
    pass


class SearchError(AskAlfredError):
    """Raised when search operation fails."""
    pass
