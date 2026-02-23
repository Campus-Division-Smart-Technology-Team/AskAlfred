#!/usr/bin/env python3
"""
Document content extraction and processing for Alfred Local Ingestion.
This module provides functions to securely fetch files, extract text from various document formats, 
chunk text for embedding, and handle specific cases like property and maintenance CSVs 
with building name normalisation and metadata extraction.
"""

from __future__ import annotations

import io
import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING, Any

import pandas as pd
import tiktoken
from docx import Document as DocxDocument
from pypdf import PdfReader
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:  # pylint: disable=broad-except
    pdfminer_extract_text = None
from building import normalise_building_name
from alfred_exceptions import ParseError, ValidationError
from interfaces import EmbeddingsResult
from maintenance_utils import normalise_priority
from config import (
    INGEST_BACKOFF_BASE,
    INGEST_BACKOFF_CAP,
    INGEST_BACKOFF_JITTER_MIN,
    INGEST_BACKOFF_JITTER_SPAN,
    INGEST_FETCH_MAX_SIZE_MB,
    INGEST_EMBED_BATCH_SIZE,
)

from .utils import validate_safe_path

if TYPE_CHECKING:
    from .context import IngestContext


def _get_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger(__name__)


def parse_pivot_header(col_str: str, is_requests: bool) -> dict[str, str]:
    """
    Parse maintenance pivot headers.

    Requests examples:
        "Asbestos Requests - Other - In progress"
        "ASU Requests - RM Priority 1 - Within 1 week - Complete"
        -> {category, priority, status}

    Jobs examples:
        "BEMS Controls Jobs - Complete"
        "ASU Jobs - In progress"
        -> {category, status}
    """
    keyword = "Requests" if is_requests else "Jobs"
    cleaned = col_str.replace(keyword, "").strip()
    parts = [p.strip() for p in re.split(r"\s*-\s*", cleaned) if p.strip()]

    if is_requests:
        if len(parts) == 1:
            return {"category": parts[0], "priority": "Unknown", "status": "Unknown"}
        if len(parts) == 2:
            return {"category": parts[0], "priority": "Unknown", "status": parts[1]}
        return {
            "category": parts[0],
            "priority": " - ".join(parts[1:-1]),
            "status": parts[-1],
        }

    if len(parts) == 1:
        return {"category": parts[0], "status": "Unknown"}
    return {"category": parts[0], "status": parts[1]}


def load_tabular_data(key: str, data: bytes) -> pd.DataFrame:
    """Load CSV or Excel data into a DataFrame based on file extension."""
    ext_ = ext(key)
    if ext_ == "csv":
        return pd.read_csv(io.BytesIO(data))
    if ext_ in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data))
    raise ParseError(f"Unsupported tabular file type: {ext_}")


def backoff_sleep(
    attempt: int,
    base: float = INGEST_BACKOFF_BASE,
    cap: float = INGEST_BACKOFF_CAP,
) -> None:
    """Exponential backoff with jitter."""
    delay = min(cap, base * (2**attempt)) * (
        INGEST_BACKOFF_JITTER_MIN + random.random() * INGEST_BACKOFF_JITTER_SPAN
    )
    time.sleep(delay)


def ext(key: str) -> str:
    """Extract file extension."""
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


def fetch_bytes_secure(
    base_path: str,
    key: str,
    max_size_mb: float = INGEST_FETCH_MAX_SIZE_MB,
    *,
    logger: logging.Logger | None = None,
) -> bytes:
    """
    Securely read file from local filesystem with size validation.
    """
    filepath = validate_safe_path(base_path, key, logger=logger)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    if max_size_mb > 0 and size_mb > max_size_mb:
        raise ValidationError(
            f"File too large: {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"
        )
    with open(filepath, "rb") as file_handle:
        return file_handle.read()


def extract_text(
    key: str,
    data: bytes,
    *,
    logger: logging.Logger | None = None,
) -> str:
    """
    Extract text from standard document formats (not CSV).
    """
    extension = ext(key)

    if extension in {"txt", "md"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    if extension == "json":
        try:
            text = data.decode("utf-8")
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:  # pylint: disable=broad-except
            return data.decode("utf-8", errors="ignore")

    log = _get_logger(logger)

    if extension == "pdf":
        text = ""
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            if text.strip():
                return text
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("PDF extract failed for %s: %s", key, ex)

        if pdfminer_extract_text is None:
            return text

        if not is_fire_risk_assessment(key, text):
            return text

        try:
            log.info("Using PDFMiner fallback for %s", key)
            text = pdfminer_extract_text(io.BytesIO(data)) or ""
            return text
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("PDFMiner extract failed for %s: %s", key, ex)
            return text

    if extension == "docx":
        try:
            doc = DocxDocument(io.BytesIO(data))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as ex:  # pylint: disable=broad-except
            log.warning("DOCX extract failed for %s: %s", key, ex)
            return ""

    return ""


def extract_text_and_chunk_in_process(
    key: str,
    data: bytes,
    *,
    embed_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[str, list[str]]:
    """
    Extract text and chunk it using a local encoder.

    This is designed to run in a process pool to avoid GIL bottlenecks.
    """
    text = extract_text(key, data)
    if not text:
        return text, []

    try:
        encoder = tiktoken.encoding_for_model(embed_model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")

    tokens = encoder.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = encoder.decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - chunk_overlap)
        if end == len(tokens):
            break

    return text, chunks


def chunk_text(ctx: "IngestContext", text: str) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    tokens = ctx.encoder.encode(text)
    chunks: list[str] = []
    start = 0
    chunk_tokens = ctx.config.chunk_tokens
    overlap = ctx.config.chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = ctx.encoder.decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == len(tokens):
            break

    return chunks


def chunk_text_generator(ctx: "IngestContext", text: str):
    """Yield overlapping chunks without storing the full chunk list."""
    tokens = ctx.encoder.encode(text)
    start = 0
    chunk_tokens = ctx.config.chunk_tokens
    overlap = ctx.config.chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = ctx.encoder.decode(tokens[start:end])
        chunk = re.sub(r"\s+\n", "\n", chunk).strip()
        if chunk:
            yield chunk
        start = max(0, end - overlap)
        if end == len(tokens):
            break


def embed_texts_batch(
    ctx: "IngestContext",
    texts: list[str],
) -> EmbeddingsResult:
    """Embed a batch of texts; retry/backoff handled by Embedder."""
    if not texts:
        return EmbeddingsResult(
            embeddings_by_index={},
            errors_by_index={},
            error_summary=None,
        )
    if getattr(ctx.config, "dry_run", False):
        return EmbeddingsResult(
            embeddings_by_index={},
            errors_by_index={},
            error_summary=None,
        )

    t_start = time.perf_counter()
    result = ctx.embedder.embed_texts(
        texts,
        model=ctx.config.embed_model,
        timeout=ctx.config.openai_timeout,
        max_batch=INGEST_EMBED_BATCH_SIZE,
    )
    elapsed = time.perf_counter() - t_start
    ctx.stats.observe_timing("embed_batch_seconds", elapsed)
    ctx.stats.increment("embed_batches_total")
    ctx.stats.increment("embed_texts_total", len(texts))
    if result.errors_by_index:
        ctx.stats.increment("embed_failed_total", len(result.errors_by_index))
    if result.retry_attempts:
        ctx.stats.increment("embed_retries_total", result.retry_attempts)
    if result.batch_reductions:
        ctx.stats.increment("embed_batch_reductions_total",
                            result.batch_reductions)
    if result.rate_limit_errors:
        ctx.stats.increment("embed_rate_limit_total", result.rate_limit_errors)
    return result


def load_building_names_with_aliases(
    ctx: "IngestContext",
    base_path: str,
    key: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Load building names and alias mappings from local property CSV/Excel.

    Returns:
        (canonical_names, name_to_canonical_map, alias_to_canonical_map)
    """
    existing_names = ctx.cache.get_name_mapping()
    existing_aliases = ctx.cache.get_alias_mapping()

    if existing_names and existing_aliases:
        canonical_names = list(set(existing_names.values()))
        return canonical_names, existing_names, existing_aliases

    try:
        data = fetch_bytes_secure(base_path, key)
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            ctx.logger.warning("No 'Property name' column in CSV")
            return [], {}, {}

        canonical_names: list[str] = []
        name_to_canonical: dict[str, str] = {}
        alias_to_canonical: dict[str, str] = {}
        metadata_cache: dict[str, dict[str, Any]] = {}

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            canonical = str(prop_name).strip()
            canonical_names.append(canonical)
            aliases = set()

            name_to_canonical[canonical.lower()] = canonical
            aliases.add(canonical)

            if pd.notna(row.get("Property names")):
                for name in str(row["Property names"]).split(";"):
                    name = name.strip()
                    if name:
                        norm = normalise_building_name(name)
                        name_to_canonical[norm] = canonical
                        alias_to_canonical[norm] = canonical
                        aliases.add(name)

            if pd.notna(row.get("Property alternative names")):
                for name in str(row["Property alternative names"]).split(";"):
                    name = name.strip()
                    if name:
                        norm = normalise_building_name(name)
                        name_to_canonical[norm] = canonical
                        alias_to_canonical[norm] = canonical
                        aliases.add(name)

            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                condensed = str(row["UsrFRACondensedPropertyName"]).strip()
                if condensed:
                    norm = normalise_building_name(condensed)
                    name_to_canonical[norm] = canonical
                    alias_to_canonical[norm] = canonical
                    aliases.add(condensed)

            building_metadata = {
                "canonical_building_name": canonical,
                "building_aliases": list(aliases),
            }

            for field in [
                "Property code",
                "Property postcode",
                "Property campus",
                "UsrFRACondensedPropertyName",
                "Property names",
                "Property alternative names",
                "Property condition",
                "Property gross area (sq m)",
                "Property net area (sq m)",
            ]:
                if field in row.index and pd.notna(row[field]):
                    building_metadata[field] = str(row[field])

            metadata_cache[canonical] = building_metadata

        ctx.cache.update_from_csv(
            name_to_canonical, alias_to_canonical, metadata_cache)

        ctx.logger.info(
            "Loaded %d canonical building names with %d total name variations",
            len(canonical_names),
            len(alias_to_canonical),
        )
        return canonical_names, name_to_canonical, alias_to_canonical

    except Exception as ex:  # pylint: disable=broad-except
        ctx.logger.warning("Failed to load building names: %s", ex)
        return [], {}, {}


def is_fire_risk_assessment(key: str, text: str = "") -> bool:
    """Determine if a document is a Fire Risk Assessment (FRA)."""
    key_lower = key.lower()
    fra_patterns = [
        r"\bfra\b",
        r"fire[\s_-]?risk[\s_-]?assessment",
        r"fire_risk",
        r"firerisk",
        r"risk[\s_-]?assessment",
        r"\boas\b",
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
            "emergency lighting",
        ]
        count = sum(
            1 for indicator in content_indicators if indicator in text_lower)
        if count >= 2:
            return True

    return False


def is_bms_document(key: str) -> bool:
    """Determine if a document is related to BMS/operational systems."""
    key_lower = key.lower()
    bms_patterns = [
        r"bms",
        r"building[\s_-]?management",
        r"hvac",
        r"mechanical[\s_-]?services",
        r"electrical[\s_-]?services",
    ]
    return any(re.search(pattern, key_lower) for pattern in bms_patterns)


def extract_text_csv_by_building_enhanced(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str],
    *,
    logger: logging.Logger | None = None,
) -> list[tuple[str, str | None, str, dict[str, Any]]]:
    """
    Extract property CSV data by building.

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples.
    """
    log = _get_logger(logger)
    try:
        df = load_tabular_data(key, data)

        if "Property name" not in df.columns:
            log.warning(
                "Column 'Property name' not found in CSV. Available: %s",
                df.columns.tolist(),
            )
            return [
                (
                    key,
                    "",
                    data.decode("utf-8", errors="ignore"),
                    {"document_type": "unknown"},
                )
            ]

        building_docs: list[tuple[str, str | None, str, dict[str, Any]]] = []

        for _, row in df.iterrows():
            prop_name = row.get("Property name")
            if pd.isna(prop_name):
                continue

            raw_prop = str(prop_name).strip()
            canonical_name = alias_to_canonical.get(
                normalise_building_name(raw_prop), raw_prop)

            building_text = f"Property: {canonical_name}\n\n"

            for col, val in row.items():
                if col != "Property name" and pd.notna(val):
                    building_text += f"{col}: {val}\n"

            extra_metadata: dict[str, Any] = {}
            aliases = [raw_prop, canonical_name]

            if pd.notna(row.get("Property names")):
                aliases.extend([n.strip()
                               for n in str(row["Property names"]).split(";")])

            if pd.notna(row.get("Property alternative names")):
                aliases.extend(
                    [n.strip()
                     for n in str(row["Property alternative names"]).split(";")]
                )

            if pd.notna(row.get("UsrFRACondensedPropertyName")):
                aliases.append(str(row["UsrFRACondensedPropertyName"]).strip())
            aliases = list(set([alias for alias in aliases if alias]))

            for field in [
                "Property code",
                "Property postcode",
                "Property campus",
                "UsrFRACondensedPropertyName",
                "Property names",
                "Property alternative names",
                "Property condition",
                "Property gross area (sq m)",
                "Property net area (sq m)",
            ]:
                if field in row.index and pd.notna(row[field]):
                    extra_metadata[field] = str(row[field])

            extra_metadata["building_aliases"] = aliases
            extra_metadata["canonical_building_name"] = canonical_name
            extra_metadata["document_type"] = "planon_data"

            building_key = f"Planon Data - {canonical_name}"
            building_docs.append(
                (building_key, canonical_name, building_text, extra_metadata))

        if not building_docs:
            log.warning(
                "No buildings found in CSV, indexing as single doc")
            return [(key, "All Properties", df.to_string(), {})]

        log.info(
            "Extracted %d buildings from property CSV with aliases",
            len(building_docs),
        )
        return building_docs

    except Exception as ex:  # pylint: disable=broad-except
        log.warning("CSV extraction failed: %s", ex)
        return [(key, "", data.decode("utf-8", errors="ignore"), {"document_type": "unknown"})]


def extract_maintenance_csv(
    key: str,
    data: bytes,
    alias_to_canonical: dict[str, str],
    *,
    logger: logging.Logger | None = None,
) -> list[tuple[str, str | None, str, dict[str, Any]]]:
    """
    Extract maintenance CSV data (Requests or Jobs) by building.

    Returns:
        List of (doc_key, canonical_name, text, extra_metadata) tuples.
    """

    def _pick_building_col(df: pd.DataFrame) -> str | None:
        candidates = ["Property name", "Buildings",
                      "Building", "Property", "Site"]
        lower_map = {column.lower().strip(): column for column in df.columns}
        for candidate in candidates:
            lower = candidate.lower().strip()
            if lower in lower_map:
                return lower_map[lower]
        return None

    log = _get_logger(logger)
    try:
        df = load_tabular_data(key, data)
        building_col = _pick_building_col(df)
        if not building_col:
            log.warning(
                "No recognised building column found in %s. Available columns: %s",
                key,
                df.columns.tolist(),
            )
            return []

        key_lower = key.lower()
        if "maintenance_jobs" in key_lower or "maintenance jobs" in key_lower:
            is_requests = False
        elif "maintenance_requests" in key_lower or "maintenance requests" in key_lower:
            is_requests = True
        else:
            has_requests = any(
                "requests" in str(col).lower() for col in df.columns if col != building_col
            )
            has_jobs = any(
                "jobs" in str(col).lower() for col in df.columns if col != building_col
            )
            is_requests = has_requests and not has_jobs

        data_type = "Maintenance Requests" if is_requests else "Maintenance Jobs"
        log.info("Processing %s with %d rows", data_type, len(df))

        building_docs: list[tuple[str, str | None, str, dict[str, Any]]] = []

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
            extra_metadata: dict[str, Any] = {
                "canonical_building_name": canonical_name,
                "document_type": doc_type,
                "data_type": data_type.lower().replace(" ", "_"),
                "file_source": key,
            }

            maintenance_metrics: dict[str, Any] = {}
            total_items = 0

            for col, val in row.items():
                if col == building_col or pd.isna(val) or val == 0:
                    continue

                col_str = str(col)
                col_lower = col_str.lower()
                if is_requests:
                    if "request" not in col_lower:
                        continue
                elif not any(token in col_lower for token in ("job", "work order")):
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

                    if is_requests:
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category].setdefault(
                            priority_label, {})
                        leaf = maintenance_metrics[category][priority_label]
                        leaf[status] = leaf.get(status, 0) + count
                        building_text += (
                            f"{category} - {priority_label} - {status}: {count}\n"
                        )
                    else:
                        maintenance_metrics.setdefault(category, {})
                        maintenance_metrics[category][status] = (
                            maintenance_metrics[category].get(
                                status, 0) + count
                        )
                        building_text += f"{category} - {status}: {count}\n"

                    total_items += count
                except Exception as error:
                    log.warning(
                        "Could not parse column '%s' with value '%s': %s",
                        col,
                        val,
                        error,
                    )

            if not maintenance_metrics:
                log.warning(
                    "No maintenance metrics extracted for %s (%s).",
                    canonical_name,
                    data_type,
                )

            metrics_json = json.dumps(maintenance_metrics)
            if len(metrics_json) > 30000:
                log.warning(
                    "Maintenance metrics too large for %s (%d bytes), truncating",
                    canonical_name,
                    len(metrics_json),
                )
                extra_metadata["maintenance_metrics"] = json.dumps(
                    {
                        "total_items": total_items,
                        "categories_count": len(maintenance_metrics),
                        "categories": list(maintenance_metrics.keys()),
                    }
                )
                building_text += "\n=== Summary ===\n"
                building_text += f"Total {data_type}: {total_items}\n"
                building_text += (
                    f"Categories with activity: {len(maintenance_metrics)}\n"
                )
                building_text += (
                    "Active categories: "
                    + ", ".join(sorted(maintenance_metrics.keys()))
                    + "\n"
                )
            else:
                extra_metadata["maintenance_metrics"] = metrics_json

            doc_key = f"{data_type} - {canonical_name}"
            building_docs.append(
                (doc_key, canonical_name, building_text, extra_metadata))

        log.info("Extracted %d buildings from %s", len(building_docs), key)
        return building_docs

    except Exception as ex:  # pylint: disable=broad-except
        log.error(
            "Maintenance CSV extraction failed for %s: %s",
            key,
            ex,
            exc_info=True,
        )
        return []
