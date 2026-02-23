"""
FRA-specific integration utilities for ingest process.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import zlib
from typing import Any, TYPE_CHECKING, cast

from redis.exceptions import RedisError

from date_utils import parse_iso_date
from alfred_exceptions import wrap_exception, IngestError, ExternalServiceError
from building.normaliser import normalise_building_name
from config import (
    FRA_RISK_ITEMS_NAMESPACE,
    PAGE_LIMIT,
    INGEST_DEDUP_FETCH_BATCH_SIZE,
)
from config.constant import FRA_PARTITION_KEY_BUCKET_SIZE
from interfaces import JobRecord
from .types import EnrichedRiskItem

if TYPE_CHECKING:
    from ingest import IngestContext

# ============================================================================
# SUPERSEDED DOCUMENT HANDLING
# ============================================================================


def _fra_partition_key(building: str, buckets: int = FRA_PARTITION_KEY_BUCKET_SIZE) -> str:
    """Generate a partition key for a building name to distribute load across index partitions."""
    normalised = building.strip().lower().encode("utf-8")
    bucket = zlib.crc32(normalised) % buckets
    return f"bkt_{bucket:02d}"


def _assessment_date_to_int(value: str | None) -> int | None:
    """Convert ISO date (YYYY-MM-DD) to an integer YYYYMMDD for numeric filtering."""
    if not value or value == "__null__":
        return None
    parsed = parse_iso_date(value)
    if not parsed:
        return None
    return parsed.year * 10000 + parsed.month * 100 + parsed.day


def mark_superseded_risk_items(ctx: IngestContext,
                               building: str,
                               new_assessment_date: str) -> list[str]:
    """Mark existing risk items for the same building with older assessment dates as superseded."""
    building_key = normalise_building_name(building)
    registry_id = f"fra_supersede:{building_key}:{new_assessment_date}"
    try:
        started = ctx.job_registry.try_start(
            job_id=registry_id,
            job_type="fra_supersession",
            status="processing",
            meta={
                "building": building,
                "assessment_date": new_assessment_date,
                "namespace": FRA_RISK_ITEMS_NAMESPACE,
            },
        )
        if not started:
            existing = None
            try:
                existing = ctx.job_registry.get(registry_id)
            except (RedisError, OSError, ValueError, TypeError) as e:
                mapped_error = wrap_exception(e)
                ctx.logger.warning(
                    "JobRegistry lookup failed for %s: %s", registry_id, mapped_error)
            if existing and existing.status == "success":
                ctx.logger.info(
                    "Supersession already recorded for %s (assessment: %s); skipping",
                    building,
                    new_assessment_date,
                )
            else:
                ctx.logger.info(
                    "Supersession already processing for %s (assessment: %s); skipping",
                    building,
                    new_assessment_date,
                )
            return []
    except (RedisError, OSError, ValueError, TypeError) as e:
        mapped_error = wrap_exception(e)
        ctx.logger.warning(
            "JobRegistry start failed for %s: %s", registry_id, mapped_error)

    try:
        existing = ctx.job_registry.get(registry_id)
    except (RedisError, OSError, ValueError, TypeError) as e:
        mapped_error = wrap_exception(e)
        ctx.logger.warning(
            "JobRegistry lookup failed for %s: %s", registry_id, mapped_error)
        existing = None
    if existing and existing.status == "success":
        ctx.logger.info(
            "Supersession already recorded for %s (assessment: %s); skipping",
            building,
            new_assessment_date,
        )
        return []

    ctx.logger.info(
        f"Checking for superseded risk items in {building} "
        f"(new assessment: {new_assessment_date})"
    )

    partition_key = _fra_partition_key(building)
    new_assessment_date_int = _assessment_date_to_int(new_assessment_date)
    filter_dict: Mapping[str, Any] = {
        "partition_key": partition_key,
        "canonical_building_name": building,
        "is_current": True,
    }
    if new_assessment_date_int is not None:
        filter_dict = {
            **filter_dict,
            "fra_assessment_date_int": {"$lt": new_assessment_date_int},
        }
    else:
        filter_dict = {
            **filter_dict,
            "fra_assessment_date": {"$lt": new_assessment_date},
        }

    updated_ids: list[str] = []
    top_k = PAGE_LIMIT

    try:
        if getattr(ctx.config, "dry_run", False):
            raise IngestError("Dry-run enabled: supersession query must not be executed")
        query_vector: list[float] | None = None
        cached = ctx.cache.get_embedding(building)
        if cached:
            query_vector = cached
        try:
            if query_vector is None:
                embed_result = ctx.embedder.embed_texts(
                    [building],
                    model=ctx.config.embed_model,
                    timeout=ctx.config.openai_timeout,
                    max_batch=1,
                )
                query_vector = embed_result.embeddings_by_index.get(0)
                if query_vector:
                    ctx.cache.set_embedding(building, query_vector)
        except Exception as embed_error:  # pylint: disable=broad-except
            ctx.logger.warning(
                "Supersession embedding failed for %s: %s; falling back to zero vector",
                building,
                embed_error,
            )

        if not query_vector:
            dim = int(getattr(ctx.config, "dimension", 0))
            if dim <= 0:
                raise IngestError("Invalid embedding dimension for supersession query")
            query_vector = [0.0] * dim
        results = ctx.index.query(
            vector=query_vector,
            namespace=FRA_RISK_ITEMS_NAMESPACE,
            filter=dict(filter_dict),
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )
    except (ExternalServiceError, IngestError, OSError, ValueError, TypeError, ConnectionError, TimeoutError, RuntimeError) as e:
        mapped_error = wrap_exception(e)
        ctx.logger.error(
            "Failed to query existing risk items: %s", mapped_error)
        try:
            ctx.job_registry.upsert(
                JobRecord(
                    job_id=registry_id,
                    job_type="fra_supersession",
                    status="failed",
                    started_at_iso=datetime.now(
                        timezone.utc).isoformat() + "Z",
                    finished_at_iso=None,
                    error=str(mapped_error),
                    meta={
                        "building": building,
                        "assessment_date": new_assessment_date,
                        "namespace": FRA_RISK_ITEMS_NAMESPACE,
                    },
                )
            )
        except (RedisError, OSError, ValueError, TypeError) as registry_error:
            mapped_registry_error = wrap_exception(registry_error)
            ctx.logger.warning(
                "JobRegistry update failed for %s: %s",
                registry_id,
                mapped_registry_error,
            )
        return []

    if results and isinstance(results, dict) and "matches" in results:
        matches = results["matches"]
    else:
        matches = getattr(results, "matches", []) if results else []

    if not matches:
        ctx.logger.info("No old risk items found to supersede")
        return []

    if len(matches) >= top_k:
        ctx.logger.warning(
            "Supersession query hit top_k=%d for %s (new assessment: %s). "
            "Results may be truncated; consider partitioning or increasing limit.",
            top_k,
            building,
            new_assessment_date,
        )
        try:
            ctx.job_registry.upsert(
                JobRecord(
                    job_id=registry_id,
                    job_type="fra_supersession",
                    status="partial",
                    started_at_iso=datetime.now(
                        timezone.utc).isoformat() + "Z",
                    finished_at_iso=None,
                    error=f"query_truncated_top_k_{top_k}",
                    meta={
                        "building": building,
                        "assessment_date": new_assessment_date,
                        "namespace": FRA_RISK_ITEMS_NAMESPACE,
                        "top_k": top_k,
                        "matches": len(matches),
                    },
                )
            )
            ctx.event_sink.emit_event(
                {
                    "event_type": "fra_supersession_query_truncated",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "building": building,
                    "assessment_date": new_assessment_date,
                    "namespace": FRA_RISK_ITEMS_NAMESPACE,
                    "top_k": top_k,
                    "matches": len(matches),
                }
            )
        except (RedisError, OSError, ValueError, TypeError) as alert_error:
            mapped_alert_error = wrap_exception(alert_error)
            ctx.logger.warning(
                "JobRegistry/alert update failed for %s: %s",
                registry_id,
                mapped_alert_error,
            )

    eligible_ids: list[str] = []
    for match in matches:
        if isinstance(match, dict):
            item_id = match.get("id")
            metadata = match.get("metadata") or {}
        else:
            item_id = getattr(match, "id", None)
            metadata = getattr(match, "metadata", None) or {}
        if not item_id:
            continue

        if metadata:
            is_current = metadata.get("is_current", True)
            superseded_by = metadata.get("superseded_by") or ""
            assessment_date = metadata.get("fra_assessment_date")
            assessment_date_int = metadata.get("fra_assessment_date_int")
            if not is_current:
                continue
            if superseded_by:
                continue
            if new_assessment_date_int is not None and assessment_date_int is not None:
                if assessment_date_int >= new_assessment_date_int:
                    continue
            elif assessment_date and assessment_date >= new_assessment_date:
                continue
        eligible_ids.append(item_id)

    if not eligible_ids:
        ctx.logger.info("No eligible risk items found to supersede")
        return []

    try:
        # Pinecone's bulk update supports filters without an id, but the type
        # stubs require `id: str`. Cast to Any to avoid a false-positive.
        index_any = cast(Any, ctx.index)
        index_any.update(
            namespace=FRA_RISK_ITEMS_NAMESPACE,
            filter=dict(filter_dict),
            set_metadata={
                "superseded_by": new_assessment_date,
                "is_current": False,
                "supersession_epoch": new_assessment_date_int or new_assessment_date,
            },
            id=None,
        )
        try:
            ctx.stats.increment("fra_supersession_bulk_updates_total")
        except Exception:
            pass
        updated_ids.extend(eligible_ids)
    except (ExternalServiceError, IngestError, OSError, ValueError, TypeError, ConnectionError, TimeoutError, RuntimeError) as e:
        mapped_error = wrap_exception(e)
        ctx.logger.warning(
            "Bulk supersession update failed; falling back to per-item updates: %s",
            mapped_error,
        )
        try:
            ctx.stats.increment("fra_supersession_bulk_updates_failed_total")
        except Exception:
            pass
        for item_id in eligible_ids:
            try:
                ctx.index.update(
                    id=item_id,
                    namespace=FRA_RISK_ITEMS_NAMESPACE,
                    set_metadata={
                        "superseded_by": new_assessment_date,
                        "is_current": False,
                        "supersession_epoch": new_assessment_date_int or new_assessment_date,
                    },
                )
                try:
                    ctx.stats.increment(
                        "fra_supersession_per_item_updates_total")
                except Exception:
                    pass
                updated_ids.append(item_id)
            except (ExternalServiceError, IngestError, OSError, ValueError, TypeError, ConnectionError, TimeoutError, RuntimeError) as per_item_error:
                mapped_per_item_error = wrap_exception(per_item_error)
                ctx.logger.error(
                    "Failed to update %s: %s", item_id, mapped_per_item_error
                )
                try:
                    ctx.stats.increment(
                        "fra_supersession_per_item_updates_failed_total")
                except Exception:
                    pass

    ctx.logger.info(f"Marked {len(updated_ids)} risk items as superseded")
    try:
        status = "success"
        error = None
        if not updated_ids and matches:
            status = "partial"
            error = "No items updated (possible concurrent supersession)"
        ctx.job_registry.upsert(
            JobRecord(
                job_id=registry_id,
                job_type="fra_supersession",
                status=status,
                started_at_iso=datetime.now(timezone.utc).isoformat() + "Z",
                finished_at_iso=datetime.now(timezone.utc).isoformat() + "Z",
                error=error,
                meta={
                    "building": building,
                    "assessment_date": new_assessment_date,
                    "namespace": FRA_RISK_ITEMS_NAMESPACE,
                    "updated_ids": updated_ids,
                },
            )
        )
    except (RedisError, OSError, ValueError, TypeError) as e:
        mapped_error = wrap_exception(e)
        ctx.logger.warning(
            "JobRegistry update failed for %s: %s", registry_id, mapped_error)
    return updated_ids


def restore_superseded_items(ctx: IngestContext, item_ids: list[str]) -> int:
    """
    Restore superseded risk items if a later operation fails.

    Args:
        ctx: IngestContext
        item_ids: List of item IDs previously marked as superseded

    Returns:
        Number of items restored
    """
    if not item_ids:
        return 0

    restored = 0
    for item_id in item_ids:
        try:
            ctx.index.update(
                id=item_id,
                set_metadata={
                    "is_current": True,
                    # Pinecone metadata values cannot be null; use empty string to clear marker.
                    "superseded_by": ""
                },
                namespace=FRA_RISK_ITEMS_NAMESPACE
            )
            restored += 1
        except (ExternalServiceError, IngestError, OSError, ValueError, TypeError, ConnectionError, TimeoutError, RuntimeError) as e:
            ctx.logger.error(f"Failed to restore {item_id}: {e}")

    total = len(item_ids)
    if restored == 0:
        ctx.logger.error(
            "Rollback failed: restored 0/%d superseded risk items", total
        )
    elif restored < total:
        ctx.logger.warning(
            "Partial rollback: restored %d/%d superseded risk items",
            restored,
            total,
        )
    else:
        ctx.logger.info("Restored %d superseded risk items", restored)
    return restored

# ============================================================================
# BUILDING EXTRACTION FROM TEXT (FRA-SPECIFIC)
# ============================================================================


def deduplicate_risk_items(
    ctx: IngestContext,
    new_items: list[EnrichedRiskItem]
) -> list[EnrichedRiskItem]:
    """Remove items that already exist in index."""
    new_ids = [item["risk_item_id"] for item in new_items]

    # Batch fetch existing
    existing_ids = set()
    batch_size = max(
        1,
        int(getattr(ctx.config, "dedup_fetch_batch", INGEST_DEDUP_FETCH_BATCH_SIZE)),
    )
    for i in range(0, len(new_ids), batch_size):
        batch = [
            item_id
            for item_id in new_ids[i:i + batch_size]
            if item_id is not None
        ]
        if batch:
            response = ctx.index.fetch(
                ids=batch, namespace=FRA_RISK_ITEMS_NAMESPACE)
            existing_ids.update(response.vectors.keys())

    # Filter out existing
    return [item for item in new_items if item["risk_item_id"] not in existing_ids]
