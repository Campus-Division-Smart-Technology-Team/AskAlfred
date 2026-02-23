#!/usr/bin/env python3
"""
Ingestion helper functions for Alfred Local.
This module provides various helper functions used during the ingestion process, including:
- Batch namespace summarisation and sampling for vector batches.
- Estimation of batch byte sizes for efficient processing.
- Extraction of layout-preserved text from PDFs for improved FRA parsing.
- Detection of rate limit errors from exceptions.
These helpers are designed to support the main ingestion workflow while keeping the code modular and maintainable.
"""
import random
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Any, TypeAlias, TYPE_CHECKING
from alfred_exceptions import (
    ValidationError,
)
if TYPE_CHECKING:
    from .context import IngestContext
from .utils import (
    validate_safe_path,
)

UpsertQueueItem: TypeAlias = tuple[list[dict[str, Any]], int, int]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _summarise_batch_namespaces(batch: list[dict[str, Any]]) -> str:
    if not batch:
        return ""
    counts: dict[str, int] = {}
    for vector in batch:
        namespace = str(vector.get("namespace") or "")
        counts[namespace] = counts.get(namespace, 0) + 1
    return ",".join(f"{ns or 'default'}:{count}" for ns, count in counts.items())


def _sample_batch(
    batch: list[dict[str, Any]],
    *,
    sample_rate: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Sample vectors from a batch at the given rate.
    If no samples are drawn, return the first vector as a fallback.
    """
    if not batch:
        return []
    sample: list[dict[str, Any]] = []
    for vector in batch:
        if random.random() < sample_rate:
            sample.append(vector)
    if not sample:
        sample = [batch[0]]
    return sample


def _estimate_batch_bytes(
    batch: list[dict[str, Any]],
    *,
    sample_rate: float = 0.1,
) -> int | None:
    if not batch:
        return None
    sample = _sample_batch(batch, sample_rate=sample_rate)
    total = 0
    for vector in sample:
        try:
            total += len(json.dumps(vector, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            total += len(str(vector))
    return int(total * (len(batch) / max(len(sample), 1)))


def _estimate_metadata_bytes_per_vector(
    batch: list[dict[str, Any]],
    *,
    sample_rate: float = 0.1,
) -> int | None:
    if not batch:
        return None
    sample = _sample_batch(batch, sample_rate=sample_rate)
    total = 0
    for vector in sample:
        metadata = vector.get("metadata") or {}
        try:
            total += len(json.dumps(metadata, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            total += len(str(metadata))
    est_total = total * (len(batch) / max(len(sample), 1))
    return int(est_total / max(len(batch), 1))


def _extract_fra_layout_text(
    ctx: "IngestContext",
    *,
    base_path: str,
    key: str,
) -> str | None:
    """
    Extract layout-preserved text for FRA parsing via pdftotext -layout.
    Falls back to None on any error, letting caller use existing text.
    """
    try:
        pdf_path = validate_safe_path(base_path, key, logger=ctx.logger)
    except ValidationError as ex:
        ctx.logger.warning("FRA layout extract skipped for %s: %s", key, ex)
        return None

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        text = tmp_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return None
        return text
    except FileNotFoundError:
        ctx.logger.info(
            "pdftotext not available; using standard PDF text for %s", key
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        ctx.logger.warning("pdftotext failed for %s: %s", key, stderr)
    except OSError as exc:
        ctx.logger.warning("pdftotext OS error for %s: %s", key, exc)
    finally:
        tmp_path.unlink(missing_ok=True)

    return None


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate-limit/throttle error."""
    error_str = str(error).lower()
    return any(marker in error_str for marker in ("429", "rate limit", "throttl"))
