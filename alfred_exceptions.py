class AskAlfredError(Exception):
    """Base exception for AskAlfred."""


class ConfigError(AskAlfredError):
    """Raised when configuration is missing or invalid."""


class IngestError(AskAlfredError):
    """Raised when ingestion pipeline fails."""


class ExternalServiceError(IngestError):
    """Raised when an external service fails (OpenAI/Pinecone)."""


class ModelNotInitialisedError(AskAlfredError):
    """Raised when a required model is not ready."""


class RoutingError(AskAlfredError):
    """Raised when routing or namespace mapping is invalid."""


class ParseError(AskAlfredError):
    """Raised when parsing fails."""


class ValidationError(IngestError):
    """Raised when validation fails."""


class EnrichmentError(AskAlfredError):
    """Raised when enrichment fails."""


class BuildingNotFoundError(AskAlfredError):
    """Raised when building cannot be found in cache."""


class HandlerError(AskAlfredError):
    """Raised when handler fails to process query."""


class SearchError(AskAlfredError):
    """Raised when search operation fails."""


class UnexpectedError(IngestError):
    """Fallback for unexpected exceptions raised during ingestion."""


class RollbackError(IngestError):
    """Raised when a rollback fails or is incomplete."""


class DeadlockError(RuntimeError):
    """Raised when FRA lock acquisition times out (possible deadlock)."""

# Retriable errors


class RetriableError(IngestError):
    pass


class NetworkError(RetriableError):
    pass


class RateLimitError(RetriableError):
    pass


class DataValidationError(IngestError):
    pass


def wrap_exception(error: Exception) -> IngestError:
    """
    Map non-alfred exceptions to suitable alfred_exceptions types.
    Use this to standardize error handling in ingestion paths.
    """
    if isinstance(error, IngestError):
        return error

    if isinstance(error, (TimeoutError, ConnectionError)):
        return ExternalServiceError(str(error))

    if isinstance(error, (ValueError, KeyError, TypeError, UnicodeError)):
        return ValidationError(str(error))

    if isinstance(error, (FileNotFoundError, PermissionError, OSError)):
        return IngestError(str(error))

    return UnexpectedError(str(error))
