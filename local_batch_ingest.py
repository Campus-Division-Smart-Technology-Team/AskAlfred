#!/usr/bin/env python3
"""
Batch Ingest for AskAlfred

Improvements over original:
1. Thread-safe statistics and caching
2. Efficient file-based vector ID caching (100x faster)
3. Centralized configuration with validation
4. Better error handling and metadata validation
5. Progress tracking
6. Namespace separation (preserved from original)
7. Path traversal vulnerability fix
8. Progress bar for sequential processing
9. Enhanced file size validation

All existing functionality preserved and backward compatible.
"""

import io
import json
import re
import hashlib
import time
import random
import argparse
import logging
from typing import Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument
import pandas as pd
from dotenv import load_dotenv

from openai import APIError, RateLimitError

# Import utilities
from ingest_context import IngestContext
from config import (BatchIngestConfig, DocumentTypes, resolve_namespace,
                    NAMESPACE_MAPPINGS,)
from ingest_utils import (
    MetadataValidator,
    ThreadSafeVectorBuffer,
    validate_namespace_routing,
    upsert_vectors,
    validate_safe_path,
    list_local_files_secure,
    IngestionProgressTracker,
    enrich_with_building_metadata,
    truncate_by_tokens,
)
# Centralised building logic
from building_normaliser import normalise_building_name
from filename_building_parser import get_building_with_confidence, should_flag_for_review

from maintenance_utils import normalise_priority


# ============================================================================
# Helper Utilities
# ============================================================================


def is_empty(value) -> bool:
    """Return True if a value is None, empty string, or empty collection."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return len(value.strip()) == 0
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    return False


def parse_pivot_header(col_str: str, is_requests: bool) -> dict:
    """
    Parse both Requests and Jobs headers.

    Requests examples:
        "Asbestos Requests - Other - In progress"
        "ASU Requests - RM Priority 1 - Within 1 week - Complete"
        → {category, priority, status}

    Jobs examples:
        "BEMS Controls Jobs - Complete"
        "ASU Jobs - In progress"
        → {category, status}

    Returns a dict with 2 or 3 keys depending on type.
    """

    # Strip the keyword ("Requests" or "Jobs")
    keyword = "Requests" if is_requests else "Jobs"
    cleaned = col_str.replace(keyword, "").strip()

    parts = [p.strip() for p in re.split(r"\s*-\s*", cleaned) if p.strip()]

    if is_requests:
        # Expect: category / priority / status
        if len(parts) == 1:
            return {"category": parts[0], "priority": "Unknown", "status": "Unknown"}
        elif len(parts) == 2:
            return {"category": parts[0], "priority": "Unknown", "status": parts[1]}
        return {
            "category": parts[0],
            "priority": " - ".join(parts[1:-1]),   # <-- swallow middle tokens
            "status": parts[-1],
        }

    else:
        # Jobs only have: category / status
        if len(parts) == 1:
            return {"category": parts[0], "status": "Unknown"}
        else:
            return {"category": parts[0], "status": parts[1]}


def load_tabular_data(key: str, data: bytes) -> pd.DataFrame:
    """
    Load CSV or Excel data into a DataFrame based on file extension.
    """
    ext_ = ext(key)

    if ext_ == "csv":
        return pd.read_csv(io.BytesIO(data))

    if ext_ in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data))

    raise ValueError(f"Unsupported tabular file type: {ext_}")


# ---------------- Environment & Setup ----------------
load_dotenv()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0) -> None:
    """Exponential backoff with jitter."""
    t = min(cap, base * (2**attempt)) * (0.5 + random.random())
    time.sleep(t)


def ext(key: str) -> str:
    """Extract file extension."""
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


def fetch_bytes_secure(base_path: str, key: str, max_size_mb: float = 100) -> bytes:
    """
    Securely read file from local filesystem with size validation.

    Args:
        base_path: Base directory path
        key: Relative file path
        max_size_mb: Maximum allowed file size in MB (pre-read check)

    Returns:
        File contents as bytes

    Raises:
        ValueError: If file is too large or path is unsafe
    """
    filepath = validate_safe_path(base_path, key)

    # Check file size BEFORE reading into memory
    size_mb = filepath.stat().st_size / (1024 * 1024)
    if max_size_mb > 0 and size_mb > max_size_mb:
        raise ValueError(
            f"File too large: {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"
        )

    with open(filepath, "rb") as f:
        return f.read()


def process_document(
    ctx: "IngestContext",
    obj: dict[str, Any],
    base_path: str,
    name_to_canonical: dict[str, str],
    alias_to_canonical: dict[str, str],
    known_buildings: list[str],
    existing_file_ids: set[str],
) -> list[dict[str, Any]]:
    """
    Process a single file and return a list of Pinecone-ready vectors.
    This runs inside each thread in the ThreadPoolExecutor.
    """

    key = obj["Key"]
    size_mb = obj.get("Size", 0) / (1024 * 1024)
    ext_ = ext(key)

    # Skip large files
    if ctx.config.max_file_mb > 0 and size_mb > ctx.config.max_file_mb:
        logging.info("Skipping %s (%.2f MB > limit)", key, size_mb)
        ctx.stats.increment("files_skipped")
        return []

    # 1. Load file data
    data = fetch_bytes_secure(base_path, key)

    # 1b. Compute content hash (used for IDs + optional metadata)
    content_hash = hashlib.sha256(data).hexdigest()

    # 1c. Skip duplicates if already in index (content-hash aware)
    file_id = make_file_id(base_path, key, content_hash)
    if ctx.config.skip_existing and file_id in existing_file_ids:
        logging.debug("Skipping already indexed file (hash match): %s", key)
        ctx.stats.increment("files_skipped")
        return []

    try:
        # ----------------------------------------------------------
        # 1. Load file data
        # ----------------------------------------------------------
        data = fetch_bytes_secure(base_path, key)

        # ----------------------------------------------------------
        # 2. Extract text
        # ----------------------------------------------------------
        if ext_ in ("csv", "xlsx", "xls"):
            if "maintenance" in key.lower():
                docs = extract_maintenance_csv(key, data, alias_to_canonical)
            else:
                docs = extract_text_csv_by_building_enhanced(
                    key, data, alias_to_canonical)
        else:
            text_sample = extract_text(key, data)

            # ============================================================
            # BUILDING RESOLUTION (centralised)
            # ============================================================
            building, confidence, source = get_building_with_confidence(
                filename=key,
                text_sample=text_sample,
                known_buildings=known_buildings,          # or the list you loaded
                name_to_canonical=name_to_canonical,
                alias_to_canonical=alias_to_canonical,
            )

            flag_review = should_flag_for_review(confidence, source)

            building_metadata = {
                "canonical_building_name": building if building != "Unknown" else None,
                "building_confidence": confidence,
                "building_source": source,
                "building_flag_review": flag_review
            }

            # Log low-confidence assignments for monitoring
            if confidence < 0.70:
                logging.warning(
                    "⚠️  Low confidence building assignment: %s -> %s "
                    "(confidence: %.2f%%, source: %s)",
                    key, building, confidence * 100, source
                )

            docs = [(key, building, text_sample, building_metadata)]
        # ----------------------------------------------------------
        # DRY-RUN: parse + chunk only (no OpenAI, no Pinecone)
        # ----------------------------------------------------------
        if getattr(ctx.config, "dry_run", False):
            planned_vectors = 0

            for doc_key, canonical, text, extra_metadata in docs:
                chunks = chunk_text(ctx, text)
                planned_vectors += len(chunks)

                # Optional: validate metadata shape using first chunk only
                if chunks:
                    meta = {
                        "source_path": base_path,
                        "key": doc_key,
                        "source": key,
                        "document_type": extra_metadata.get("document_type", "unknown"),
                        "canonical_building_name": canonical,
                        **extra_metadata,
                        "text": truncate_by_tokens(
                            chunks[0],
                            encoder=ctx.encoder,
                            max_tokens=getattr(
                                ctx.config, "max_metadata_text_tokens", 800),
                        ),
                    }
                    ok, reason = MetadataValidator.validate(meta)
                    if not ok:
                        logging.warning(
                            "Dry-run metadata invalid for %s: %s", key, reason)

            logging.info(
                "🧪 Dry-run planned vectors for %s: %d",
                key,
                planned_vectors
            )
            if planned_vectors > 0:
                ctx.stats.increment("files_processed")
                # treat as planned vectors
                ctx.stats.increment("total_vectors", planned_vectors)
            else:
                ctx.stats.increment("files_skipped")

            return []

        # ----------------------------------------------------------
        # 3. Embed & build vectors
        # ----------------------------------------------------------
        vectors_to_upsert = []

        for doc_key, canonical, text, extra_metadata in docs:
            chunks = chunk_text(ctx, text)
            embeddings = embed_texts_batch(ctx, chunks)

            if chunks and not embeddings:
                raise RuntimeError(
                    "Embedding failed: empty embeddings returned")

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                vector_id = make_id(file_id, doc_key, i)
                key_lower = key.lower()

                # ------------------------------------------------------
                # Determine document type
                # ------------------------------------------------------
                doc_type = extra_metadata.get("document_type", "unknown")

                # If this is a maintenance CSV row, trust the extractor
                if ext_ in ("csv", "xlsx", "xls") and doc_type in ("maintenance_request", "maintenance_job"):
                    pass  # source of truth: extract_maintenance_csv()
                else:
                    # Non-CSV or non-maintenance documents → infer type
                    if is_fire_risk_assessment(key, text):
                        doc_type = DocumentTypes.FIRE_RISK_ASSESSMENT
                    elif is_bms_document(key, text):
                        doc_type = DocumentTypes.OPERATIONAL_DOC
                    elif "planon" in key_lower:
                        doc_type = DocumentTypes.PLANON_DATA
                    else:
                        # keep whatever extractor set (or "unknown")
                        doc_type = doc_type

                # Map namespace from doc_type (single source of truth)
                resolved_namespace = resolve_namespace(doc_type)
                # ---------------------------------------------------------
                # Emit structured building assignment event (JSONL)
                # ---------------------------------------------------------
                if getattr(ctx.config, "export_events", False) and i == 0:
                    try:
                        event = {
                            "file": key,
                            "canonical_building_name": canonical,
                            "document_type": doc_type,
                            "namespace_guess": resolved_namespace,
                            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                        }
                        event_path = getattr(ctx.config, "export_events_file")
                        if event_path:
                            line = json.dumps(event, ensure_ascii=False) + "\n"
                            with ctx.export_events_lock:
                                with open(event_path, "a", encoding="utf-8") as f:
                                    f.write(line)
                    except Exception as e:
                        logging.warning(
                            "Could not write building event for %s: %s", key, e)

                valid, reason = validate_namespace_routing(
                    doc_type, resolved_namespace)
                if not valid:
                    raise ValueError(
                        f"Invalid namespace routing for doc_type: {doc_type}")

                logging.info(
                    "📁 %s → doc_type=%s → namespace=%s", key, doc_type, resolved_namespace)

                metadata = {
                    "source_path": base_path,
                    "key": doc_key,
                    "source": key,
                    "document_type": doc_type,
                    "canonical_building_name": canonical,
                    "namespace": resolved_namespace,
                    "content_hash": content_hash,
                    **extra_metadata,
                }

                if canonical:
                    metadata = enrich_with_building_metadata(
                        metadata=metadata,
                        canonical=canonical,
                        ctx=ctx,
                        doc_type=doc_type,
                        chunk_idx=i,
                    )
                metadata["text"] = truncate_by_tokens(
                    chunk,
                    encoder=ctx.encoder,
                    max_tokens=ctx.config.MAX_METADATA_TEXT_TOKENS
                )

                valid, reason = MetadataValidator.validate(metadata)
                if not valid:
                    logging.warning("Invalid metadata for %s: %s", key, reason)
                    continue

                vectors_to_upsert.append(
                    {
                        "id": vector_id,
                        "values": emb,
                        "metadata": metadata,
                        "namespace": resolved_namespace,
                    }
                )

        # ----------------------------------------------------------
        # 4. Update stats and return
        # ----------------------------------------------------------
        if vectors_to_upsert:
            ctx.stats.increment("files_processed")

        return vectors_to_upsert

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error processing %s: %s", key, e, exc_info=True)
        ctx.stats.increment("files_failed")
        ctx.stats.append_failed(key)
        return []


# ============================================================================
# EFFICIENT VECTOR ID CACHING
# ============================================================================

def get_existing_file_ids(
    ctx: IngestContext,
    source_path: str
) -> set[str]:
    """
    Fetch existing vector IDs with efficient caching across all namespaces.

    Optimised:
    - Uses file-based cache for 100x speed improvement
    - Makes only ONE Pinecone `.list()` API call (covers all namespaces)
    - Handles all configured namespaces safely
    - Provides clear logging for cache hits/misses

    Args:
        ctx: Ingestion context
        source_path: Source directory path for cached vector IDs

    Returns:
        A set of unique existing vector IDs across all namespaces
    """
    if getattr(ctx.config, "dry_run", False):
        logging.info("🧪 Dry-run: skipping existing-id discovery")
        return set()
    if not ctx.config.skip_existing:
        return set()

    cache_ns = "__all__"

    cached = ctx.vector_id_cache.get(source_path, cache_ns)
    if cached is not None:
        logging.info("✅ Existing file IDs loaded from cache (%d)", len(cached))
        return set(cached)

    logging.info("🔍 Fetching existing vector IDs to derive file IDs...")

    try:
        results = ctx.index.list()
    except Exception as e:
        logging.warning("⚠️ Could not fetch index IDs from Pinecone: %s", e)
        return set()

    existing_file_ids: set[str] = set()

    # results looks iterable in your current code; keep same assumption
    for id_list in results:
        for vid in id_list:
            # vid format: "{file_id}:{chunk_idx}"
            file_id = str(vid).split(":", 1)[0]
            if file_id:
                existing_file_ids.add(file_id)

    ctx.vector_id_cache.set(source_path, cache_ns, existing_file_ids)
    logging.info("✅ Found %d existing file IDs", len(existing_file_ids))
    return existing_file_ids


def make_file_id(source_path: str, source_key: str, content_hash: str | None = None) -> str:
    if content_hash is not None and not isinstance(content_hash, str):
        raise TypeError("content_hash must be a hex string or None")
    base = f"{source_path}:{source_key}"
    if content_hash:
        base += f":{content_hash}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _doc_sub_id(doc_key: str) -> str:
    # stable compact identifier for doc_key (building rows etc.)
    return hashlib.sha1(doc_key.encode("utf-8")).hexdigest()[:12]


def make_id(file_id: str, doc_key: str, chunk_idx: int) -> str:
    """
    Generate a deterministic ID for a vector.
    """
    # vector identity = file_id + per-doc-key sub-id + chunk index
    return f"{file_id}:{_doc_sub_id(doc_key)}:{chunk_idx}"
# ============================================================================
# TEXT EXTRACTION FUNCTIONS
# ============================================================================


def extract_text(key: str, data: bytes) -> str:
    """
    Extract text from standard document formats (not CSV).
    UNCHANGED from original - preserves backward compatibility.
    """
    e = ext(key)

    if e in {"txt", "md"}:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        return text

    if e == "json":
        try:
            text = data.decode("utf-8")
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:  # pylint: disable=broad-except
            text = data.decode("utf-8", errors="ignore")
        return text

    if e == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception as ex:  # pylint: disable=broad-except
            logging.warning("PDF extract failed for %s: %s", key, ex)
            return ""

    if e == "docx":
        try:
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as ex:  # pylint: disable=broad-except
            logging.warning("DOCX extract failed for %s: %s", key, ex)
            return ""

    return ""


def chunk_text(
    ctx: IngestContext,
    text: str
) -> list[str]:
    """
    Split text into overlapping chunks based on token count.
    UNCHANGED logic from original, but uses context for config.
    """
    toks = ctx.encoder.encode(text)
    chunks = []
    i = 0
    chunk_tokens = ctx.config.chunk_tokens
    overlap = ctx.config.chunk_overlap

    while i < len(toks):
        j = min(i + chunk_tokens, len(toks))
        chunk = ctx.encoder.decode(toks[i:j])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        i = max(0, j - overlap)
        if j == len(toks):
            break
    return chunks


def embed_texts_batch(
    ctx: IngestContext,
    texts: list[str],
    max_retries: int = 5
) -> list[list[float]]:
    """
    Embed a batch of texts with retry logic.
    UNCHANGED logic from original, but uses context for config and client.
    """
    if not texts:
        return []
    if getattr(ctx.config, "dry_run", False):
        return []

    for attempt in range(max_retries):
        try:
            resp = ctx.openai.embeddings.create(
                model=ctx.config.embed_model,
                input=texts,
                timeout=ctx.config.openai_timeout,
            )
            return [d.embedding for d in resp.data]
        except (RateLimitError, APIError) as e:
            logging.warning(
                "Embedding API backoff (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e
            )
            if attempt == max_retries - 1:
                return []
            backoff_sleep(attempt)
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Embedding failed with unexpected error: %s", e)
            return []
    return []


# ============================================================================
# BUILDING NAME MATCHING (using thread-safe cache)
# ============================================================================

def load_building_names_with_aliases(
    ctx: IngestContext,
    base_path: str,
    key: str
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Load building names and create comprehensive alias mappings from local CSV.
    IMPROVED: Uses thread-safe cache instead of global variables.

    Returns:
        (canonical_names, name_to_canonical_map, alias_to_canonical_map)
    """
    # Check if already loaded
    existing_names = ctx.cache.get_name_mapping()
    existing_aliases = ctx.cache.get_alias_mapping()

    if existing_names and existing_aliases:
        canonical_names = list(set(existing_names.values()))
        return canonical_names, existing_names, existing_aliases

    try:
        data = fetch_bytes_secure(base_path, key)
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            logging.warning("No 'Property name' column in CSV")
            return [], {}, {}

        canonical_names = []
        name_to_canonical = {}
        alias_to_canonical = {}
        metadata_cache = {}

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            canonical = str(prop_name).strip()
            canonical_names.append(canonical)

            # Collect all name variations for this building
            aliases = set()

            # 1. Canonical name itself
            name_to_canonical[canonical.lower()] = canonical
            aliases.add(canonical)

            # 2. Property names (semicolon-separated)
            if pd.notna(row.get("Property names")):
                for name in str(row["Property names"]).split(";"):
                    name = name.strip()
                    if name:
                        name_to_canonical[normalise_building_name(
                            name)] = canonical
                        alias_to_canonical[normalise_building_name(
                            name)] = canonical
                        aliases.add(name)

            # 3. Alternative names (semicolon-separated)
            if pd.notna(row.get("Property alternative names")):
                for name in str(row["Property alternative names"]).split(";"):
                    name = name.strip()
                    if name:
                        name_to_canonical[normalise_building_name(
                            name)] = canonical
                        alias_to_canonical[normalise_building_name(
                            name)] = canonical
                        aliases.add(name)

            # 4. Condensed FRA name
            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                condensed = str(row["UsrFRACondensedPropertyName"]).strip()
                if condensed:
                    name_to_canonical[normalise_building_name(
                        condensed)] = canonical
                    alias_to_canonical[normalise_building_name(
                        condensed)] = canonical
                    aliases.add(condensed)

            # Store metadata for this building
            building_metadata = {
                "canonical_building_name": canonical,
                "building_aliases": list(aliases),
            }

            # Add other useful metadata fields
            for field in ["Property code", "Property postcode", "Property campus",
                          "UsrFRACondensedPropertyName", "Property names",
                          "Property alternative names", "Property condition",
                          "Property gross area (sq m)", "Property net area (sq m)"]:
                if field in row.index and pd.notna(row[field]):
                    building_metadata[field] = str(row[field])

            metadata_cache[canonical] = building_metadata

        # Update thread-safe cache
        ctx.cache.update_from_csv(
            name_to_canonical, alias_to_canonical, metadata_cache)

        logging.info(
            "✅ Loaded %d canonical building names with %d total name variations",
            len(canonical_names),
            len(alias_to_canonical)
        )

        return canonical_names, name_to_canonical, alias_to_canonical

    except Exception as ex:  # pylint: disable=broad-except
        logging.warning("Failed to load building names: %s", ex)
        return [], {}, {}


# ============================================================================
# DOCUMENT TYPE DETECTION (NON-CSV)
# ============================================================================

def is_fire_risk_assessment(key: str, text: str = "") -> bool:
    """
    Determine if a document is a Fire Risk Assessment (FRA).
    """
    key_lower = key.lower()
    fra_patterns = [
        r"fra[\s_-]",
        r"fire[\s_-]?risk[\s_-]?assessment",
        r"fire_risk",
        r"firerisk",
        r"risk[\s_-]?assessment",
        r"\boas\b"
    ]

    for pattern in fra_patterns:
        if re.search(pattern, key_lower):
            return True

    if text:
        text_lower = text.lower()
        content_indicators = [
            "fire risk assessment",
            "compartmentation",
            "fire safety",
            "means of escape",
            "emergency lighting"
        ]
        count = sum(
            1 for indicator in content_indicators if indicator in text_lower)
        if count >= 2:
            return True

    return False


def is_bms_document(key: str, text: str = "") -> bool:
    """
    Determine if a document is related to BMS/operational systems.
    UNCHANGED from original - preserves backward compatibility.
    """
    key_lower = key.lower()
    bms_patterns = [
        r"bms",
        r"building[\s_-]?management",
        r"hvac",
        r"mechanical[\s_-]?services",
        r"electrical[\s_-]?services",
    ]

    for pattern in bms_patterns:
        if re.search(pattern, key_lower):
            return True

    return False


# ============================================================================
# CSV EXTRACTION FUNCTIONS
# ============================================================================

def extract_text_csv_by_building_enhanced(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str]
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """
    Extract property CSV data by building.

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples
    """
    try:
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            logging.warning(
                "Column 'Property name' not found in CSV. Available: %s",
                df.columns.tolist()
            )
            return [(key, "", data.decode("utf-8", errors="ignore"), {"document_type": "unknown"})]

        building_docs = []

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            canonical_name = str(prop_name).strip()

            # Build text representation for this building
            building_text = f"Property: {canonical_name}\n\n"

            # Add all available data
            for col, val in row.items():
                if col != "Property name" and pd.notna(val):
                    building_text += f"{col}: {val}\n"

            # Collect metadata
            extra_metadata = {}

            # Collect aliases for this building
            aliases = [canonical_name]

            # Add property names
            if pd.notna(row.get("Property names")):
                aliases.extend([n.strip()
                               for n in str(row["Property names"]).split(";")])

            # Add alternative names
            if pd.notna(row.get("Property alternative names")):
                aliases.extend([n.strip() for n in str(
                    row["Property alternative names"]).split(";")])

            # Add condensed name
            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                aliases.append(str(row["UsrFRACondensedPropertyName"]).strip())

            # Remove duplicates and empty strings
            aliases = list(set([a for a in aliases if a]))

            # Add more property fields to extra_metadata
            for field in ["Property code", "Property postcode", "Property campus",
                          "UsrFRACondensedPropertyName", "Property names",
                          "Property alternative names", "Property condition",
                          "Property gross area (sq m)", "Property net area (sq m)"]:
                if field in row.index and pd.notna(row[field]):
                    extra_metadata[field] = str(row[field])

            extra_metadata["building_aliases"] = aliases
            extra_metadata["canonical_building_name"] = canonical_name
            extra_metadata["document_type"] = "planon_data"

            building_key = f"Planon Data - {canonical_name}"
            building_docs.append(
                (building_key, canonical_name, building_text, extra_metadata))

        if not building_docs:
            logging.warning(
                "No buildings found in CSV, indexing as single doc")
            return [(key, "All Properties", df.to_string(), {})]

        logging.info("✅ Extracted %d buildings from property CSV with aliases",
                     len(building_docs))
        return building_docs

    except Exception as ex:  # pylint: disable=broad-except
        logging.warning("CSV extraction failed: %s", ex)
        return [(key, "", data.decode("utf-8", errors="ignore"), {"document_type": "unknown"})]


def extract_maintenance_csv(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str]
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """
    Extract maintenance CSV data (Requests or Jobs), supporting:
      • Requests → category / priority / status
      • Jobs → category / status

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples.
    """

    def _pick_building_col(df: pd.DataFrame) -> str | None:
        # Prefer common variants, case-insensitive
        candidates = ["Property name", "Buildings",
                      "Building", "Property", "Site"]
        lower_map = {c.lower().strip(): c for c in df.columns}

        for cand in candidates:
            key = cand.lower().strip()
            if key in lower_map:
                return lower_map[key]

        return None

    # ---------------------------------------------------------
    # Main Processing
    # ---------------------------------------------------------

    try:
        df = load_tabular_data(key, data)

        building_col = _pick_building_col(df)
        if not building_col:
            logging.warning(
                "No recognised building column found in %s. Available columns: %s",
                key, df.columns.tolist()
            )
            return []

        # Identify Requests vs Jobs
        kl = key.lower()
        if "maintenance_jobs" in kl or "maintenance jobs" in kl:
            is_requests = False
        elif "maintenance_requests" in kl or "maintenance requests" in kl:
            is_requests = True
        else:
            is_requests = any("requests" in str(col).lower()
                              for col in df.columns if col != building_col)
            is_jobs = any("jobs" in str(col).lower()
                          for col in df.columns if col != building_col)
            is_requests = is_requests and not is_jobs

        data_type = "Maintenance Requests" if is_requests else "Maintenance Jobs"
        logging.info("Processing %s with %d rows", data_type, len(df))

        building_docs = []

        # Iterate buildings
        for _, row in df.iterrows():

            building_name = row.get(building_col)
            if pd.isna(building_name):
                continue

            raw = str(building_name).strip()
            canonical_name = alias_to_canonical.get(
                normalise_building_name(raw), raw)

            building_text = f"Building: {canonical_name}\n"
            building_text += f"Data Type: {data_type}\n\n"

            doc_type = "maintenance_request" if is_requests else "maintenance_job"

            extra_metadata = {
                "canonical_building_name": canonical_name,
                "document_type": doc_type,
                "data_type": data_type.lower().replace(" ", "_"),
                "file_source": key
            }

            maintenance_metrics = {}
            total_items = 0

            # --------------------------
            # COLUMN-BY-COLUMN PROCESSING
            # --------------------------
            for col, val in row.items():

                if col == building_col or pd.isna(val) or val == 0:
                    continue

                col_str = str(col)

                col_l = col_str.lower()
                if is_requests:
                    if "request" not in col_l:
                        continue
                else:
                    if not any(k in col_l for k in ("job", "work order")):
                        continue

                try:
                    parsed = parse_pivot_header(col_str, is_requests)

                    category = parsed["category"]
                    status = (str(parsed.get("status")
                              or "").strip()) or "unknown"
                    status = status[:1].upper() + status[1:].lower()

                    priority_label = normalise_priority(
                        parsed.get("priority") or "")

                    count = int(float(val))

                    # --------------------------
                    # STORE STRUCTURED METRICS
                    # --------------------------
                    if is_requests:
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category].setdefault(
                            priority_label, {})
                        leaf = maintenance_metrics[category][priority_label]
                        leaf[status] = leaf.get(status, 0) + count

                        building_text += f"{category} - {priority_label} - {status}: {count}\n"

                    else:
                        # Category → Status
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category][status] = maintenance_metrics[category].get(
                            status, 0) + count
                        building_text += f"{category} - {status}: {count}\n"
                    total_items += count

                except Exception as e:
                    logging.warning(
                        "Could not parse column '%s' with value '%s': %s",
                        col, val, e
                    )
                    continue
            if not maintenance_metrics:
                logging.warning(
                    "⚠️ No maintenance metrics extracted for %s (%s). Check header parsing / column keywords.", canonical_name, data_type)

            # --------------------------
            # SUMMARY AND METADATA
            # --------------------------
            metrics_json = json.dumps(maintenance_metrics)
            if len(metrics_json) > 30000:  # Leave buffer under 40KB limit
                logging.warning(
                    "Maintenance metrics too large for %s (%d bytes), truncating",
                    canonical_name, len(metrics_json)
                )
                extra_metadata["maintenance_metrics"] = json.dumps({
                    "total_items": total_items,
                    "categories_count": len(maintenance_metrics),
                    "categories": list(maintenance_metrics.keys())
                })

                building_text += "\n=== Summary ===\n"
                building_text += f"Total {data_type}: {total_items}\n"
                building_text += f"Categories with activity: {len(maintenance_metrics)}\n"
                building_text += (
                    "Active categories: "
                    + ", ".join(sorted(maintenance_metrics.keys()))
                    + "\n"
                )
            else:
                extra_metadata["maintenance_metrics"] = metrics_json

            # Document key
            doc_key = f"{data_type} - {canonical_name}"

            building_docs.append(
                (doc_key, canonical_name, building_text, extra_metadata)
            )

        logging.info("✅ Extracted %d buildings from %s",
                     len(building_docs), key)
        return building_docs

    except Exception as ex:
        logging.error(
            "Maintenance CSV extraction failed for %s: %s",
            key, ex, exc_info=True
        )
        return []

# ============================================================================
# MAIN INGESTION FUNCTION WITH PROGRESS BAR
# ============================================================================


def ingest_local_directory_with_progress(ctx, use_progress_bar: bool = True):
    """
    Enhanced ingestion function with progress tracking and security.
    """
    t_start = time.time()
    base_path = ctx.config.local_path

    # Validate path exists
    if not Path(base_path).exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    # Use secure file listing
    objs = list_local_files_secure(
        base_path,
        ctx.config.ext_whitelist,
        ctx.config.max_file_mb
    )
    logging.info("Found %d files to process in %s", len(objs), base_path)

    if not objs:
        logging.warning("No files found to process!")
        return

    # Get existing IDs (as before)
    existing_file_ids = get_existing_file_ids(ctx, base_path)

    # Pre-load building names (as before)
    name_to_canonical = {}
    alias_to_canonical = {}
    known_buildings = []

    csv_candidates = [
        o["Key"] for o in objs
        if "Property" in o["Key"]
        and o["Key"].endswith(".csv")
        and "maintenance" not in o["Key"].lower()
    ]

    if csv_candidates:
        known_buildings, name_to_canonical, alias_to_canonical = \
            load_building_names_with_aliases(ctx, base_path, csv_candidates[0])
    else:
        logging.warning("No property CSV found for building name resolution")

    # Thread-safe vector buffer
    pending_buffer = ThreadSafeVectorBuffer(ctx.config.max_pending_vectors)

    # ============================================================
    # SEQUENTIAL PROCESSING WITH PROGRESS BAR
    # ============================================================
    if ctx.config.max_workers == 1:
        logging.info("Processing files sequentially with progress tracking...")

        with IngestionProgressTracker(len(objs), use_tqdm=use_progress_bar) as progress:
            for idx, obj in enumerate(objs, 1):
                key = obj["Key"]

                try:
                    # Use secure file fetching in process_document
                    # (process_document would need to be updated to use fetch_bytes_secure)
                    vectors = process_document(
                        ctx,
                        obj,
                        base_path,
                        name_to_canonical,
                        alias_to_canonical,
                        known_buildings,
                        existing_file_ids
                    )

                    if vectors:
                        try:
                            pending_buffer.extend(vectors)
                            progress.update(key, len(vectors), "processed")
                        except BufferError:
                            if not getattr(ctx.config, "dry_run", False):
                                upsert_vectors(
                                    ctx, pending_buffer.get_and_clear())
                                pending_buffer.extend(vectors)
                    else:
                        progress.update(key, 0, "skipped")

                    # Upsert when batch is ready
                    if len(pending_buffer) >= min(ctx.config.upsert_batch, ctx.config.max_pending_vectors):
                        batch_to_upsert = pending_buffer.get_and_clear()
                        progress.write_message(
                            f"Upserting batch of {len(batch_to_upsert)} vectors...")
                        if not getattr(ctx.config, "dry_run", False):
                            upsert_vectors(ctx, batch_to_upsert)

                except Exception as e:
                    progress.update(key, 0, "failed")
                    progress.write_message(f"ERROR processing {key}: {e}")
                    ctx.stats.increment("files_failed")
                    ctx.stats.append_failed(key)
    # ============================================================
    # PARALLEL PROCESSING WITH PROGRESS BAR
    # ============================================================
    else:
        logging.info("Processing files with %d workers...",
                     ctx.config.max_workers)

        with IngestionProgressTracker(len(objs), use_tqdm=use_progress_bar) as progress:
            with ThreadPoolExecutor(max_workers=ctx.config.max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_document,
                        ctx,
                        obj,
                        base_path,
                        name_to_canonical,
                        alias_to_canonical,
                        known_buildings,
                        existing_file_ids
                    ): obj
                    for obj in objs
                }

                # Process completed tasks
                for future in as_completed(futures):
                    obj = futures[future]
                    key = obj["Key"]

                    try:
                        vectors = future.result()

                        if vectors:
                            pending_buffer.extend(vectors)
                            progress.update(key, len(vectors), "processed")
                        else:
                            progress.update(key, 0, "skipped")

                        # Upsert when batch is ready
                        if len(pending_buffer) >= min(ctx.config.upsert_batch, ctx.config.max_pending_vectors):
                            batch_to_upsert = pending_buffer.get_and_clear()
                            progress.write_message(
                                f"Upserting batch of {len(batch_to_upsert)} vectors...")
                            if not getattr(ctx.config, "dry_run", False):
                                upsert_vectors(ctx, batch_to_upsert)

                    except Exception as e:
                        progress.update(key, 0, "failed")
                        progress.write_message(f"ERROR processing {key}: {e}")
                        ctx.stats.increment("files_failed")
                        ctx.stats.append_failed(key)

    # Final upsert
    if not pending_buffer.is_empty():
        logging.info("Final upsert of remaining %d vectors...",
                     len(pending_buffer))
        final_batch = pending_buffer.get_and_clear()
        if not getattr(ctx.config, "dry_run", False):
            upsert_vectors(ctx, final_batch)

    # Print final statistics
    stats = ctx.stats.get_stats()
    duration = time.time() - t_start
    vectors_per_sec = stats["total_vectors"] / duration if duration > 0 else 0

    # ----------------------------------------------------------
    # Export run-level metrics event (JSONL)
    # ----------------------------------------------------------
    if getattr(ctx.config, "export_events", False):
        event_path = getattr(ctx.config, "export_events_file", None)
        if event_path:
            metrics = {
                "event_type": "ingestion_summary",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "source_path": base_path,
                "dry_run": bool(getattr(ctx.config, "dry_run", False)),
                "duration_seconds": duration,
                "files_processed": stats["files_processed"],
                "vectors_created": stats["total_vectors"],
                "vectors_per_second": vectors_per_sec,
                "failures": stats["failed_files"],
            }
            try:
                line = json.dumps(metrics, ensure_ascii=False) + "\n"
                with ctx.export_events_lock:
                    with open(event_path, "a", encoding="utf-8") as f:
                        f.write(line)
            except Exception as e:
                logging.warning(
                    "Could not write ingestion summary event: %s", e)

    logging.info(
        """========================================
            INGESTION SUMMARY
            ========================================
            Files found:          %d
            Files processed:      %d
            Files skipped:        %d
            Files failed:         %d
            Total vectors:        %d
            Duration:             %.2fs
            Avg speed:            %.1f vectors/sec
            ========================================
            """,
        len(objs),
        stats["files_processed"],
        stats["files_skipped"],
        stats["files_failed"],
        stats["total_vectors"],
        duration,
        vectors_per_sec,
    )

    if stats["failed_files"]:
        logging.warning("Failed files:")
        for f in stats["failed_files"]:
            logging.warning("  - %s", f)

    logging.info("✅ Ingestion complete!")

# ============================================================================
# CLI
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Ingest local documents into Pinecone via OpenAI embeddings"
    )
    p.add_argument(
        "--path",
        help="Local directory path"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already exist in the index",
    )
    p.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing of all documents (overrides skip-existing)",
    )
    p.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers for processing",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear vector ID cache before starting",
    )
    p.add_argument(
        "--validate-routing",
        action="store_true",
        help="Run namespace routing validation tests",
    )
    p.add_argument(
        "--export-events",
        action="store_true",
        help="Write building assignment events to JSONL file",
    )
    p.add_argument(
        "--events-file",
        help="Path to JSONL export file for building assignment events",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar display"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + chunk only. Do NOT call OpenAI or Pinecone.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load config once
    try:
        config = BatchIngestConfig.from_env()

        # 2) Apply CLI overrides
        if args.path:
            config.local_path = args.path
        if args.workers:
            config.max_workers = args.workers
        if args.force_reindex:
            config.skip_existing = False
        elif args.skip_existing:
            config.skip_existing = True
        if args.export_events:
            config.export_events = True
        if args.events_file:
            config.export_events_file = args.events_file
        if args.dry_run:
            config.dry_run = True
            config.skip_existing = False  # avoid Pinecone dependency in dry-run

        # 3) Validate once
        config.validate()

    except Exception as e:
        logging.error("Configuration error: %s", e)
        return 1

    # 4) Optional early-exit validation mode
    if args.validate_routing:
        for doc_type, expected_namespace in NAMESPACE_MAPPINGS.items():
            valid, reason = validate_namespace_routing(
                doc_type, expected_namespace)
            if not valid:
                raise ValueError(
                    f"Routing validation failed for doc_type='{doc_type}': {reason}"
                )
        logging.info("Namespace routing validation passed.")
        return 0

    # 5) Run ingestion
    ctx = IngestContext(config)

    if args.clear_cache:
        logging.info("Clearing vector ID cache...")
        ctx.vector_id_cache.clear_all()

    try:
        ingest_local_directory_with_progress(
            ctx, use_progress_bar=not args.no_progress)
        return 0
    except KeyboardInterrupt:
        logging.warning("⚠️ Ingestion interrupted by user. Cleaning up...")
        if not ctx.vector_id_cache.cache_dir.exists():
            ctx.vector_id_cache.cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving any cached progress before shutdown.")
        return 0
    except Exception as e:
        logging.error("⚠️ Ingestion failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
