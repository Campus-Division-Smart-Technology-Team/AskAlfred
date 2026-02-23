#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinecone utilities for index management and search operations.
"""

from typing import Any, Optional
import json
import logging
from clients import get_pc, get_oai
from config import normalise_ns


def list_index_names() -> list[str]:
    """Get list of available Pinecone index names."""
    try:
        pc = get_pc()
        idxs = pc.list_indexes()
    except Exception:  # pylint: disable=broad-except
        return []
    if hasattr(idxs, "names"):
        return list(idxs.names())
    if isinstance(idxs, dict) and "indexes" in idxs:
        return [i["name"] for i in idxs["indexes"]]
    return list(idxs) if isinstance(idxs, (list, tuple)) else []


def open_index(name: str):
    """Open and return a Pinecone index."""
    pc = get_pc()
    return pc.Index(name)


def list_namespaces_for_index(idx) -> list[Optional[str]]:
    """
    Return available namespaces for an index.
    None represents the default namespace.
    """
    try:
        stats = idx.describe_index_stats()
        ns_dict = (stats or {}).get("namespaces") or {}

        # Get namespace names, converting empty strings to None
        names = []
        for key in ns_dict.keys():
            if key == "" or key is None:
                names.append(None)
            elif isinstance(key, str):
                names.append(key)

        # If no namespaces found, include default
        if not names:
            names.append(None)

        # Remove duplicates and sort (None first)
        unique_names = list(set(names))
        unique_names.sort(key=lambda x: (x is not None, x or ""))

        return unique_names
    except Exception:  # pylint: disable=broad-except
        return [None]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    oai = get_oai()
    res = oai.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def vector_query(idx: Any, namespace: Optional[str], query: str, k: int, embed_model: str, metadata_filter: Optional[dict] = None) -> dict[str, Any]:
    """
    Perform vector search using client-side embeddings.

    Args:
        idx: Pinecone index
        namespace: Namespace to query (None for default namespace)
        query: Search query text
        k: Number of results to return
        embed_model: Embedding model to use
        metadata_filter: Optional metadata filter for the query

    Returns:
        Query results dictionary
    """
    vec = embed_texts([query], embed_model)[0]

    # Build query parameters, only include namespace if not None
    query_params = {
        'vector': vec,
        'top_k': k,
        'include_metadata': True
    }

    namespace = normalise_ns(namespace)
    if namespace is not None:
        query_params['namespace'] = namespace

    if metadata_filter:
        query_params['filter'] = metadata_filter

    return idx.query(**query_params)


def query_all_chunks(
    index: Any,
    namespace: Optional[str],
    query_vector: list[float],
    filter_dict: Optional[dict[str, Any]] = None,
    top_k: int = 1000,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """
    Query Pinecone index and return all matches.

    This function performs a single query to retrieve up to top_k results.
    For very large datasets, consider implementing pagination.

    Args:
        index: Pinecone index object
        namespace: Namespace to query (None for default namespace)
        query_vector: Query vector (typically a zero vector for fetching all)
        filter_dict: Optional metadata filter
        top_k: Maximum number of results to return
        include_metadata: Whether to include metadata in results

    Returns:
        list of match dictionaries with id, score, and metadata
    """
    try:
        # Build query parameters
        query_params = {
            'vector': query_vector,
            'top_k': min(top_k, 10000),  # Pinecone max limit
            'include_metadata': include_metadata
        }
        namespace = normalise_ns(namespace)
        if namespace is not None:
            query_params['namespace'] = namespace

        if filter_dict:
            query_params['filter'] = filter_dict

        # Execute query
        response = index.query(**query_params)

        # Extract matches
        if hasattr(response, 'matches'):
            matches = response.matches
        elif isinstance(response, dict) and 'matches' in response:
            matches = response['matches']
        else:
            logging.warning("Unexpected response format from Pinecone query")
            return []

        # Convert to list of dicts
        results = []
        for match in matches:
            if hasattr(match, 'to_dict'):
                match_dict = match.to_dict()
            elif isinstance(match, dict):
                match_dict = match
            else:
                match_dict = {
                    'id': getattr(match, 'id', None),
                    'score': getattr(match, 'score', 0.0),
                    'metadata': getattr(match, 'metadata', {})
                }
            results.append(match_dict)

        logging.info("Retrieved %d matches from index", len(results))
        return results

    except Exception as e:
        logging.error("Error in query_all_chunks: %s", e)
        return []


def _as_dict(obj: Any) -> dict[str, Any]:
    """Convert object to dictionary if possible, else return empty dict."""
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, dict):
                return result
        except (TypeError, AttributeError):
            pass
    if isinstance(obj, dict):
        return obj
    return {}


def safe_metadata(m):
    if isinstance(m, dict):
        return (m.get("metadata")
                or m.get("values")
                or m.get("fields")
                or {})
    if hasattr(m, "metadata"):
        return m.metadata or {}
    if hasattr(m, "values"):
        return m.values or {}
    return {}


NULL_SENTINEL = "__null__"


def desanitise_metadata_from_pinecone(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert Pinecone-safe sentinel values back to None."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value == NULL_SENTINEL:
            clean[key] = None
            continue
        if isinstance(value, list):
            items = []
            for item in value:
                if item == NULL_SENTINEL:
                    items.append(None)
                else:
                    items.append(item)
            clean[key] = items
            continue
        clean[key] = value
    return clean


def normalise_matches(raw: Any) -> list[dict[str, Any]]:
    """Normalise Pinecone results from either `matches` or `result.hits` shapes."""
    data = _as_dict(raw)

    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        out: list[dict[str, Any]] = []
        for m in data["matches"]:
            md = desanitise_metadata_from_pinecone(safe_metadata(m))
            out.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": md or {},
                "text": (md or {}).get("text") or (md or {}).get("content") or (md or {}).get("chunk") or (
                    md or {}).get("body") or "",
                "source": (md or {}).get("source") or (md or {}).get("url") or (md or {}).get("doc") or "",
                # Extract key from metadata
                "key": (md or {}).get("key") or "",
                # Skip publication_date from metadata as it's misleading
            })
        return out

    hits = (data.get("result") or {}).get(
        "hits") if isinstance(data, dict) else []
    if isinstance(hits, list) and hits:
        out = []
        for h in hits:
            fields = desanitise_metadata_from_pinecone(
                h.get("fields") or h.get("metadata") or {}
            )
            text_val = fields.get("text") or fields.get(
                "content") or fields.get("chunk") or fields.get("body") or ""
            out.append({
                "id": h.get("_id"),
                "score": h.get("_score"),
                "metadata": fields,
                "text": text_val,
                "source": fields.get("source") or fields.get("url") or fields.get("doc") or "",
                "key": fields.get("key") or "",  # Extract key from metadata
                # Skip publication_date from metadata
            })
        return out

    return []


def sanitise_metadata_for_pinecone(metadata: dict[str, Any]) -> dict[str, Any]:
    """Sanitize metadata to Pinecone-compatible types."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = NULL_SENTINEL
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
            continue
        if isinstance(value, list):
            items = []
            for item in value:
                if item is None:
                    items.append(NULL_SENTINEL)
                    continue
                if isinstance(item, str):
                    items.append(item)
                else:
                    items.append(str(item))
            clean[key] = items
            continue
        if isinstance(value, dict):
            try:
                clean[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                clean[key] = str(value)
            continue
        clean[key] = str(value)
    return clean
