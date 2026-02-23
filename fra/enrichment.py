"""
FRA Enrichment

Holds non-triage enrichment logic for FRA risk items.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, date, timezone
from typing import Optional
from config import (DocumentTypes, FRA_RISK_ITEMS_NAMESPACE, RISK_LEVEL_MAP,)
from .types import CompletionStatus, RiskItem


class FRAEnricher:
    """Enrich raw risk items with derived metadata (not triage flags)."""

    def compute_completion_status(
        self,
        *,
        expected_date: Optional[date],
        completion_date: Optional[date],
        today: date,
        data_quality_issues: Optional[list[str]] = None,
    ) -> CompletionStatus:
        """Compute current completion status using dates and today's date."""
        if completion_date:
            if expected_date and completion_date > expected_date:
                if data_quality_issues is not None:
                    data_quality_issues.append("completed_after_due_date")
            return "complete"
        if expected_date and expected_date < today:
            return "overdue"
        return "open"

    def enrich_base_fields(
        self,
        *,
        risk_item: RiskItem,
        risk_level: int,
    ) -> dict[str, object]:
        """
        Fill derived metadata fields on a risk item.

        Returns:
            Dict of fields to merge into the enriched item.
        """
        risk_level_text = risk_item.get("risk_level_text") or RISK_LEVEL_MAP.get(
            str(risk_level), "Unknown"
        )

        risk_item_id = risk_item.get("risk_item_id") or self._generate_risk_item_id(
            risk_item.get("canonical_building_name"),
            risk_item.get("fra_assessment_date"),
            risk_item.get("issue_number"),
        )

        ingestion_timestamp = risk_item.get(
            "ingestion_timestamp") or datetime.now(timezone.utc).isoformat()

        document_type = risk_item.get(
            "document_type") or DocumentTypes.FRA_RISK_ITEM
        namespace = risk_item.get("namespace") or FRA_RISK_ITEMS_NAMESPACE
        is_current = risk_item.get("is_current")
        if is_current is None:
            is_current = True
        superseded_by = risk_item.get("superseded_by")

        return {
            "risk_item_id": risk_item_id,
            "risk_level_text": risk_level_text,
            "ingestion_timestamp": ingestion_timestamp,
            "document_type": document_type,
            "namespace": namespace,
            "is_current": is_current,
            "superseded_by": superseded_by,
        }

    def _generate_risk_item_id(
        self,
        building: Optional[str],
        assessment_date: Optional[str],
        issue_num: Optional[str]
    ) -> str:
        """Generate stable hash identifier for risk item."""
        building_safe = re.sub(r"[^a-zA-Z0-9]+", "_",
                               building or "unknown").lower()
        date_part = assessment_date or "unknown"
        issue_part = issue_num or "unknown"

        raw_id = f"{building_safe}|{date_part}|{issue_part}"
        raw_id = raw_id.strip()
        if not raw_id:
            raw_id = "unknown|unknown|unknown"
        if len(raw_id) > 512:
            raw_id = raw_id[:512]
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:32]
        return f"risk_{digest}"
