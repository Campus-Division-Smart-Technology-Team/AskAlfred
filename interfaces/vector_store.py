# VectorStore port
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, Optional

from config import normalise_ns
from alfred_exceptions import ExternalServiceError


class VectorStore(Protocol):
    def upsert(self, vectors: list[dict[str, Any]], namespace: Optional[str] = None) -> Any: ...
    def fetch(self, ids: list[str], namespace: Optional[str] = None) -> Any: ...
    def query(self, **kwargs: Any) -> Any: ...
    def list_prefix(self, prefix: str, namespace: Optional[str] = None) -> Iterable[list[str]]: ...
    def list(self, namespace: Optional[str] = None) -> Iterable[list[str]]: ...


class PineconeVectorStore:
    """Thin wrapper around a Pinecone index with namespace normalization."""

    def __init__(self, index: Any):
        self._index = index

    def upsert(self, vectors: list[dict[str, Any]], namespace: Optional[str] = None) -> Any:
        try:
            return self._index.upsert(vectors=vectors, namespace=normalise_ns(namespace))
        except Exception as error:  # pylint: disable=broad-except
            raise ExternalServiceError(str(error)) from error

    def fetch(self, ids: list[str], namespace: Optional[str] = None) -> Any:
        try:
            return self._index.fetch(ids=ids, namespace=normalise_ns(namespace))
        except Exception as error:  # pylint: disable=broad-except
            raise ExternalServiceError(str(error)) from error

    def query(self, **kwargs: Any) -> Any:
        try:
            if "namespace" in kwargs:
                kwargs["namespace"] = normalise_ns(kwargs.get("namespace"))
            return self._index.query(**kwargs)
        except Exception as error:  # pylint: disable=broad-except
            raise ExternalServiceError(str(error)) from error

    def list_prefix(self, prefix: str, namespace: Optional[str] = None) -> Iterable[list[str]]:
        try:
            return self._index.list(prefix=prefix, namespace=normalise_ns(namespace))
        except Exception as error:  # pylint: disable=broad-except
            raise ExternalServiceError(str(error)) from error

    def list(self, namespace: Optional[str] = None) -> Iterable[list[str]]:
        try:
            return self._index.list(namespace=normalise_ns(namespace))
        except Exception as error:  # pylint: disable=broad-except
            raise ExternalServiceError(str(error)) from error


__all__ = [
    "VectorStore",
    "PineconeVectorStore",
]
