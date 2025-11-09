#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-safe ingestion utilities and metadata validation for AskAlfred.
Fixed version: adds complete ThreadSafeVectorBuffer and validate_namespace_routing.
"""

import json
import logging
import time
import pickle
import hashlib
from threading import Lock, RLock
from typing import Dict, Any, Optional, Set, Tuple, List
from pathlib import Path


# ---------------------------------------------------------------------------
# THREAD-SAFE STATS
# ---------------------------------------------------------------------------

class ThreadSafeStats:
    def __init__(self):
        self._lock = Lock()
        self._stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "total_vectors": 0,
            "vectors_skipped": 0,
            "failed_files": [],
        }

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    def append_failed(self, filename: str) -> None:
        with self._lock:
            self._stats["failed_files"].append(filename)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._stats.copy()


# ---------------------------------------------------------------------------
# THREAD-SAFE CACHE
# ---------------------------------------------------------------------------

class ThreadSafeCache:
    def __init__(self):
        self._lock = RLock()
        self._name_cache: Dict[str, str] = {}
        self._alias_cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    def update_from_csv(
        self,
        name_to_canonical: Dict[str, str],
        alias_to_canonical: Dict[str, str],
        metadata_cache: Dict[str, Dict[str, Any]],
    ) -> None:
        with self._lock:
            self._name_cache.update(name_to_canonical)
            self._alias_cache.update(alias_to_canonical)
            for k, v in metadata_cache.items():
                self._metadata_cache[k] = v.copy()

    def get_name_mapping(self) -> Dict[str, str]:
        with self._lock:
            return self._name_cache.copy()

    def get_alias_mapping(self) -> Dict[str, str]:
        with self._lock:
            return self._alias_cache.copy()

    def get_metadata(self, building_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached metadata for a building.
        Fixed: Added missing implementation.

        Args:
            building_name: The canonical building name

        Returns:
            Dictionary of cached metadata or None if not found
        """
        with self._lock:
            metadata = self._metadata_cache.get(building_name)
            # Return a copy to prevent external modifications
            return metadata.copy() if metadata else None

    def set_metadata(self, building_name: str, metadata: Dict[str, Any]) -> None:
        """
        Cache metadata for a building.

        Args:
            building_name: The canonical building name
            metadata: Metadata dictionary to cache
        """
        with self._lock:
            self._metadata_cache[building_name] = metadata.copy()

    def has_metadata(self, building_name: str) -> bool:
        """Check if metadata exists for a building."""
        with self._lock:
            return building_name in self._metadata_cache


# ---------------------------------------------------------------------------
# VECTOR ID CACHE
# ---------------------------------------------------------------------------

class VectorIDCache:
    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()

    def _get_cache_path(self, source_path: str, namespace: Optional[str]) -> Path:
        ns = namespace or "default"
        h = hashlib.md5(source_path.encode()).hexdigest()
        return self.cache_dir / f"{ns}_{h}.pkl"

    def get(self, source_path: str, namespace: Optional[str]) -> Optional[Set[str]]:
        file = self._get_cache_path(source_path, namespace)
        with self._lock:
            if not file.exists():
                return None
            if time.time() - file.stat().st_mtime > self.ttl_seconds:
                return None
            try:
                with open(file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None

    def set(self, source_path: str, namespace: Optional[str], ids: Set[str]) -> None:
        file = self._get_cache_path(source_path, namespace)
        with self._lock:
            with open(file, "wb") as f:
                pickle.dump(ids, f)

    def clear_all(self) -> None:
        """Clear all cached vector ID files."""
        with self._lock:
            if not self.cache_dir.exists():
                return
            try:
                for file in self.cache_dir.glob("*.pkl"):
                    file.unlink(missing_ok=True)
                logging.info(
                    "🧹 Cleared all vector ID cache files from %s", self.cache_dir)
            except Exception as e:
                logging.warning("Failed to clear vector ID cache: %s", e)


# ---------------------------------------------------------------------------
# METADATA VALIDATOR
# ---------------------------------------------------------------------------

class MetadataValidator:
    REQUIRED_FIELDS = {"source_path", "key", "source", "text", "document_type"}
    MAX_METADATA_SIZE = 40960
    MAX_TEXT_LENGTH = 1000

    @staticmethod
    def validate(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        missing = MetadataValidator.REQUIRED_FIELDS - metadata.keys()
        if missing:
            return False, f"Missing required fields: {missing}"

        nulls = [k for k, v in metadata.items() if v is None]
        if nulls:
            return False, f"Null values in {nulls}"

        try:
            size = len(json.dumps(metadata))
            if size > MetadataValidator.MAX_METADATA_SIZE:
                return False, "Metadata too large"
        except Exception as e:
            return False, f"Serialization error: {e}"

        return True, None


# ---------------------------------------------------------------------------
# THREAD-SAFE VECTOR BUFFER
# ---------------------------------------------------------------------------

class ThreadSafeVectorBuffer:
    """Thread-safe buffer for pending vectors."""

    def __init__(self, max_size: int = 10000):
        self._lock = RLock()
        self._buffer: List[Dict[str, Any]] = []
        self.max_size = max_size

    def add(self, vector: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._buffer) >= self.max_size:
                raise BufferError("Vector buffer full")
            self._buffer.append(vector)

    def get_and_clear(self) -> List[Dict[str, Any]]:
        """Retrieve and clear all buffered vectors (used for upserts)."""
        with self._lock:
            data = self._buffer[:]
            self._buffer.clear()
            return data

    def extend(self, vectors: List[Dict[str, Any]]) -> None:
        """Add multiple vectors to the buffer."""
        with self._lock:
            remaining_space = self.max_size - len(self._buffer)
            if len(vectors) > remaining_space:
                raise BufferError("Not enough space in buffer for extend()")
            self._buffer.extend(vectors)

    def size(self) -> int:
        """Return the current size of the buffer."""
        with self._lock:
            return len(self._buffer)

    def __len__(self) -> int:
        """Return the number of vectors in the buffer (len() support)."""
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        """Return True if the buffer has no vectors."""
        with self._lock:
            return len(self._buffer) == 0


# ---------------------------------------------------------------------------
# VALIDATE NAMESPACE ROUTING
# ---------------------------------------------------------------------------

def validate_namespace_routing(namespace: Optional[str]) -> bool:
    """Simple namespace validation helper used by batch_ingest."""
    allowed = {None, "planon_data", "fire_risk_assessments", "operational_docs",
               "maintenance_requests", "maintenance_jobs"}
    return namespace in allowed

# ---------------------------------------------------------------------------
# UPSERT VECTORS
# ---------------------------------------------------------------------------


def upsert_vectors(ctx, vectors: List[Dict[str, Any]]) -> None:
    """
    Upsert vectors to Pinecone index in batches.
    Expects each vector dict to have 'id', 'values', 'metadata', and 'namespace'.
    """
    if not vectors:
        return

    try:
        index = ctx.index
        grouped: Dict[Optional[str], List[Dict[str, Any]]] = {}

        # Group by namespace
        for v in vectors:
            ns = v.get("namespace")
            grouped.setdefault(ns, []).append(v)

        # Upsert per namespace
        from math import ceil

        MAX_REQUEST_VECTORS = 300  # ✅ Safe under 2 MB limit for 1536-dim embeddings

        for ns, vecs in grouped.items():
            clean_vecs = [{k: v for k, v in v.items() if k != "namespace"}
                          for v in vecs]

            # Split large batches into smaller chunks
            total = len(clean_vecs)
            chunks = ceil(total / MAX_REQUEST_VECTORS)
            for i in range(chunks):
                start = i * MAX_REQUEST_VECTORS
                end = start + MAX_REQUEST_VECTORS
                batch = clean_vecs[start:end]
                try:
                    index.upsert(vectors=batch, namespace=ns)
                    logging.info(
                        "Upserted %d/%d vectors into namespace '%s'",
                        len(batch), total, ns or "__default__"
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(
                        "Error during batch upsert (%d-%d): %s", start, end, e)

        ctx.stats.increment("total_vectors", len(vectors))

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error during upsert: %s", e, exc_info=True)
        ctx.stats.increment("files_failed")
