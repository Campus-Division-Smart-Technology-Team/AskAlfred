"""
FRA Triage Computer

Computes deterministic triage flags for fire risk assessment items.
Provides decision support through structured prioritisation, NOT automated decisions.
"""

import logging
from typing import Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from date_utils import parse_iso_date
from config import (
    FRA_LONG_OVERDUE_DAYS,
    FRA_CRITICAL_OVERDUE_DAYS,
    FRA_EXTREME_OVERDUE_DAYS,
    FRA_MAX_DAYS_SANITY,
    FRA_RISK_BASE_SCORES,
    FRA_OVERDUE_DIVISOR_DAYS,
    FRA_OVERDUE_MULTIPLIER_CAP,
    NO_JOB_REF_SCORE_MULTIPLIER,
    FRA_NO_JOB_REF_MIN_RISK_LEVEL,
    FRA_RISK_SCORE_MAX,
    FRA_PRIORITY_HIGH_RISK_LEVEL,
    FRA_PRIORITY_MEDIUM_RISK_LEVEL,
    RISK_LEVEL_MAP,
)
from .types import RiskItem, EnrichedRiskItem
from .enrichment import FRAEnricher


@dataclass
class TriageConfig:
    long_overdue_days: int = FRA_LONG_OVERDUE_DAYS
    critical_overdue_days: int = FRA_CRITICAL_OVERDUE_DAYS
    extreme_overdue_days: int = FRA_EXTREME_OVERDUE_DAYS
    max_days_sanity: int = FRA_MAX_DAYS_SANITY
    risk_base_scores: dict[int, int] = field(
        default_factory=lambda: dict(FRA_RISK_BASE_SCORES)
    )
    overdue_divisor_days: int = FRA_OVERDUE_DIVISOR_DAYS
    overdue_multiplier_cap: float = FRA_OVERDUE_MULTIPLIER_CAP
    no_job_ref_score_multiplier: float = NO_JOB_REF_SCORE_MULTIPLIER
    no_job_ref_min_risk_level: int = FRA_NO_JOB_REF_MIN_RISK_LEVEL
    risk_score_max: int = FRA_RISK_SCORE_MAX
    priority_high_risk_level: int = FRA_PRIORITY_HIGH_RISK_LEVEL
    priority_medium_risk_level: int = FRA_PRIORITY_MEDIUM_RISK_LEVEL
    risk_level_map: dict[str, str] = field(
        default_factory=lambda: dict(RISK_LEVEL_MAP)
    )


class FRATriageComputer:
    """
    Compute deterministic triage flags for risk items.

    All flags are based on objective criteria and are fully auditable.
    This system provides decision support only - never makes fire safety decisions.
    """

    # Risk category keywords
    CATEGORY_KEYWORDS = {
        "structural": [
            "door", "compartment", "wall", "ceiling", "staircase",
            "floor", "construction", "fire-resisting", "partition"
        ],
        "electrical": [
            "electrical", "wiring", "distribution board", "pat",
            "cable", "socket", "lighting", "power"
        ],
        "escape_routes": [
            "escape", "exit", "signage", "travel distance", "corridor",
            "evacuation", "assembly point", "means of escape"
        ],
        "detection_systems": [
            "alarm", "detection", "emergency lighting", "suppression",
            "extinguisher", "fire fighting", "ansul", "sprinkler"
        ],
        "housekeeping": [
            "storage", "combustible", "housekeeping", "fire load",
            "waste", "clutter", "accumulation"
        ],
        "procedural": [
            "training", "policy", "peep", "drill", "procedure",
            "evacuation plan", "fire warden", "management"
        ]
    }

    def __init__(
        self,
        verbose: bool = False,
        config: Optional[TriageConfig] = None,
        enricher: Optional[FRAEnricher] = None,
    ):
        """
        Initialise triage computer.

        Args:
            verbose: Enable detailed logging
        """
        self.config = config or TriageConfig()
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.enricher = enricher or FRAEnricher()

    def _safe_int(self, v: Any, default: int = 0) -> int:
        try:
            return int(str(v).strip())
        except (TypeError, ValueError, AttributeError):
            return default

    def compute_flags(self, risk_item: RiskItem) -> EnrichedRiskItem:
        """
        Add triage flags to a risk item record.

        Args:
            risk_item: Structured risk item from parser

        Returns:
            Risk item with added triage flags (modifies in place and returns)
        """
        today = datetime.now(timezone.utc).date()

        # Parse dates
        assessment_date = parse_iso_date(
            risk_item.get("fra_assessment_date"))
        expected_date = parse_iso_date(
            risk_item.get("expected_completion_date"))
        completion_date = parse_iso_date(
            risk_item.get("actual_completion_date"))

        # Compute time-based metrics
        days_since_assessment = (
            today - assessment_date).days if assessment_date else None

        days_until_due = None
        days_overdue = None
        if expected_date:
            days_until_due = (expected_date - today).days
            if days_until_due < 0:
                days_overdue = abs(days_until_due)

        # Extract key attributes
        risk_level_raw = self._safe_int(risk_item.get("risk_level"))

        # ====================================================================
        # INPUT VALIDATION / DATA QUALITY
        # ====================================================================

        data_quality_issues: list[str] = []
        item_label = f"{risk_item.get('canonical_building_name', 'Unknown')}#{risk_item.get('issue_number', '?')}"

        # Compute completion status here to ensure it's based on current date.
        completion_status = self.enricher.compute_completion_status(
            expected_date=expected_date,
            completion_date=completion_date,
            today=today,
            data_quality_issues=data_quality_issues,
        )
        is_complete = completion_status == "complete"
        has_job_ref = bool(risk_item.get("job_reference"))

        # Risk level should be 1-5. Treat out-of-range as invalid for scoring/flags.
        if risk_level_raw < 1 or risk_level_raw > 5:
            data_quality_issues.append("risk_level_out_of_range")
            self.logger.warning(
                "DATA QUALITY: %s - risk_level out of range: %s",
                item_label,
                risk_level_raw,
            )
            risk_level = 0
        else:
            risk_level = risk_level_raw

        # Date consistency checks
        if assessment_date and assessment_date > today:
            data_quality_issues.append("assessment_date_in_future")
            self.logger.warning(
                "DATA QUALITY: %s - assessment_date in future: %s",
                item_label,
                assessment_date,
            )

        if expected_date and assessment_date and expected_date < assessment_date:
            data_quality_issues.append("expected_date_before_assessment")
            self.logger.warning(
                "DATA QUALITY: %s - expected_date before assessment_date: %s < %s",
                item_label,
                expected_date,
                assessment_date,
            )

        if completion_date and assessment_date and completion_date < assessment_date:
            data_quality_issues.append("completion_date_before_assessment")
            self.logger.warning(
                "DATA QUALITY: %s - completion_date before assessment_date: %s < %s",
                item_label,
                completion_date,
                assessment_date,
            )

        # Extreme value checks (likely data quality issues)
        if days_since_assessment is not None and days_since_assessment > self.config.max_days_sanity:
            data_quality_issues.append("days_since_assessment_extreme")
            self.logger.warning(
                "DATA QUALITY: %s - days_since_assessment extreme: %s",
                item_label,
                days_since_assessment,
            )

        if days_until_due is not None and days_until_due > self.config.max_days_sanity:
            data_quality_issues.append("days_until_due_extreme")
            self.logger.warning(
                "DATA QUALITY: %s - days_until_due extreme: %s",
                item_label,
                days_until_due,
            )

        if days_overdue is not None and days_overdue > self.config.extreme_overdue_days:
            data_quality_issues.append("days_overdue_extreme")
            self.logger.warning(
                "DATA QUALITY: %s - days_overdue extreme: %s",
                item_label,
                days_overdue,
            )

        # ====================================================================
        # ENRICHED BASE FIELDS
        # ====================================================================

        base_fields = self.enricher.enrich_base_fields(
            risk_item=risk_item,
            risk_level=risk_level,
        )

        # ====================================================================
        # PRIMARY FLAGS
        # ====================================================================

        flag_overdue = days_overdue is not None and days_overdue > 0 and not is_complete

        flag_high_risk_no_job = (
            risk_level >= 4 and
            not has_job_ref and
            not is_complete
        )

        flag_intolerable = risk_level == 5 and not is_complete

        flag_long_overdue = (
            flag_overdue and
            days_overdue is not None and
            days_overdue > self.config.long_overdue_days
        )

        flag_critical_overdue = (
            flag_overdue and
            days_overdue is not None and
            days_overdue > self.config.critical_overdue_days
        )

        # ====================================================================
        # COMPOSITE FLAGS
        # ====================================================================

        # Requires immediate action if:
        # 1. Intolerable risk (level 5) AND not complete
        # 2. High risk (level 4+) with no job ref AND overdue
        # 3. Any risk overdue > 180 days
        requires_immediate_action = (
            flag_intolerable or
            (risk_level >= 4 and not has_job_ref and flag_overdue) or
            flag_critical_overdue
        )

        # Requires attention (lower urgency)
        requires_attention = (
            (risk_level >= 3 and flag_overdue and not requires_immediate_action) or
            flag_high_risk_no_job
        )

        # ====================================================================
        # CATEGORISATION
        # ====================================================================

        risk_category = self._categorise_risk(
            risk_item.get("issue_description"),
            risk_item.get("proposed_solution")
        )

        # ====================================================================
        # RISK SCORE
        # ====================================================================

        # Compute numeric risk score for ranking (0-100)
        risk_score = self._compute_risk_score(
            risk_level=risk_level,
            days_overdue=days_overdue,
            has_job_ref=has_job_ref,
            is_complete=is_complete
        )

        # ====================================================================
        # UPDATE RISK ITEM
        # ====================================================================

        enriched_item: EnrichedRiskItem = {  # type: ignore[assignment]
            **risk_item,
            # Time metrics
            "days_since_assessment": days_since_assessment,
            "days_until_due": days_until_due,
            "days_overdue": days_overdue,

            # Base metadata
            **base_fields,

            # Primary flags
            "flag_overdue": flag_overdue,
            "flag_high_risk_no_job": flag_high_risk_no_job,
            "flag_intolerable": flag_intolerable,
            "flag_long_overdue": flag_long_overdue,
            "flag_critical_overdue": flag_critical_overdue,

            # Unprefixed aliases for reporting compatibility
            "overdue": flag_overdue,
            "high_risk_no_job": flag_high_risk_no_job,
            "intolerable": flag_intolerable,
            "long_overdue": flag_long_overdue,
            "critical_overdue": flag_critical_overdue,

            # Composite flags
            "requires_immediate_action": requires_immediate_action,
            "requires_attention": requires_attention,

            # Categorisation
            "risk_category": risk_category,

            # Ranking
            "risk_score": risk_score,

            # Data quality
            "flag_data_quality_issue": bool(data_quality_issues),
            "data_quality_issues": data_quality_issues,

            # Current status (computed in triage)
            "completion_status": completion_status,
        }

        # Log high-priority items
        if requires_immediate_action and self.verbose:
            job_ref_text = "No job ref" if not has_job_ref else f"Job: {enriched_item['job_reference']}"
            risk_level_text = (
                enriched_item.get("risk_level_text")
                or self.config.risk_level_map.get(str(risk_level), "Unknown")
            )
            self.logger.warning(
                "🚨 IMMEDIATE ACTION REQUIRED: %s Item #%s - %s (%s)",
                enriched_item["canonical_building_name"],
                enriched_item["issue_number"],
                risk_level_text,
                job_ref_text,
            )

        return enriched_item

    def _categorise_risk(self, description: Any | None, solution: Any | None) -> str:
        """
        Categorise risk based on keywords.

        Categories:
        - structural: compartmentation, fire doors, walls
        - electrical: wiring, distribution boards, PAT
        - escape_routes: exits, signage, travel distance
        - detection_systems: alarms, emergency lighting
        - housekeeping: storage, combustibles
        - procedural: training, policies, PEEPs
        - other: doesn't match any category
        """
        description_text = str(description or "")
        solution_text = str(solution or "")
        text = f"{description_text} {solution_text}".lower()

        # Score each category
        scores: dict[str, int] = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            scores[category] = sum(1 for kw in keywords if kw in text)

        # Return highest scoring category
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=lambda category: scores[category])

        return "other"

    def _compute_risk_score(
        self,
        risk_level: int,
        days_overdue: Optional[int],
        has_job_ref: bool,
        is_complete: bool
    ) -> int:
        """
        Compute numeric risk score (0-100) for ranking.

        Formula:
        - Base score from risk level: 20 * risk_level
        - Overdue penalty: min(days_overdue / 10, 20)
        - No job ref penalty: multiplicative factor
        - Complete items: score = 0

        Returns:
            Score from 0 (lowest risk) to 100 (highest risk)
        """
        if is_complete:
            return 0

        # Exponential base score (1->10, 2->20, 3->40, 4->70, 5->100)
        score = self.config.risk_base_scores.get(risk_level, 0)

        # Overdue multiplier (up to 1.5x)
        if days_overdue:
            multiplier = 1.0 + \
                min(days_overdue / self.config.overdue_divisor_days,
                    self.config.overdue_multiplier_cap)
            score *= multiplier

        # No job penalty (10% reduction to priority)
        if not has_job_ref and risk_level >= self.config.no_job_ref_min_risk_level:
            score *= self.config.no_job_ref_score_multiplier

        return min(int(score), self.config.risk_score_max)

    def get_priority_label(self, risk_item: EnrichedRiskItem) -> str:
        """
        Get human-readable priority label for UI display.

        Returns:
            One of: "CRITICAL", "URGENT", "HIGH", "MEDIUM", "LOW", "COMPLETE"
        """
        if str(risk_item.get("completion_status", "")).lower() == "complete":
            return "COMPLETE"

        if risk_item.get("requires_immediate_action"):
            return "CRITICAL"

        if risk_item.get("requires_attention"):
            return "URGENT"

        if risk_item.get("risk_level", 0) >= self.config.priority_high_risk_level:
            return "HIGH"

        if risk_item.get("risk_level", 0) == self.config.priority_medium_risk_level:
            return "MEDIUM"

        return "LOW"

    def get_priority_emoji(self, risk_item: EnrichedRiskItem) -> str:
        """Get emoji for priority visualisation."""
        label = self.get_priority_label(risk_item)

        emoji_map = {
            "CRITICAL": "🚨",
            "URGENT": "⚠️",
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "COMPLETE": "✅"
        }

        return emoji_map.get(label, "⚪")


class FRATriageReporter:
    """Generate summary reports from triage data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_building_summary(
        self,
        building_name: str,
        risk_items: list[EnrichedRiskItem]
    ) -> dict[str, Any]:
        """
        Generate summary statistics for a building.

        Args:
            building_name: Canonical building name
            risk_items: List of risk items for this building

        Returns:
            Dictionary with summary statistics
        """
        if not risk_items:
            return {
                "building": building_name,
                "total_items": 0,
                "error": "No risk items found"
            }

        # Filter to current items only
        current_items = [
            item for item in risk_items
            if item.get("is_current", True)
        ]

        summary = {
            "building": building_name,
            "total_items": len(current_items),
            "assessment_date": current_items[0].get("fra_assessment_date") if current_items else None,

            # By status
            "by_status": {
                "complete": 0,
                "open": 0,
                "overdue": 0
            },

            # By risk level
            "by_risk_level": {
                1: 0,  # Trivial
                2: 0,  # Tolerable
                3: 0,  # Moderate
                4: 0,  # Substantial
                5: 0   # Intolerable
            },

            # By category
            "by_category": {},

            # Flags
            "flags": {
                "overdue": 0,
                "high_risk_no_job": 0,
                "intolerable": 0,
                "long_overdue": 0,
                "critical_overdue": 0,
                "requires_immediate_action": 0,
                "requires_attention": 0,
            },

            # Risk score stats
            "risk_score_avg": 0,
            "risk_score_max": 0,
        }

        # Aggregate stats
        total_score = 0

        for item in current_items:
            # Status
            status = item["completion_status"]
            summary["by_status"][status] = summary["by_status"].get(
                status, 0) + 1

            # Risk level
            level = item["risk_level"]
            summary["by_risk_level"][level] += 1

            # Category
            category = item.get("risk_category", "other")
            summary["by_category"][category] = summary["by_category"].get(
                category, 0) + 1

            # Flags
            for flag in summary["flags"]:
                if item.get(flag):
                    summary["flags"][flag] += 1

            # Risk score
            score = item.get("risk_score", 0)
            total_score += score
            summary["risk_score_max"] = max(summary["risk_score_max"], score)

        # Average risk score
        if current_items:
            summary["risk_score_avg"] = total_score / len(current_items)

        return summary

    def format_summary_text(self, summary: dict[str, Any]) -> str:
        """Format summary as readable text."""

        text = f"""
=== Fire Risk Assessment Summary ===
Building: {summary['building']}
Assessment Date: {summary['assessment_date'] or 'Unknown'}
Total Items: {summary['total_items']}

Status Breakdown:
  ✅ Complete: {summary['by_status']['complete']}
  🟢 Open: {summary['by_status']['open']}
  🔴 Overdue: {summary['by_status']['overdue']}

Risk Level Breakdown:
  🔴 Intolerable (5): {summary['by_risk_level'][5]}
  🟠 Substantial (4): {summary['by_risk_level'][4]}
  🟡 Moderate (3): {summary['by_risk_level'][3]}
  🟢 Tolerable (2): {summary['by_risk_level'][2]}
  ⚪ Trivial (1): {summary['by_risk_level'][1]}

Priority Flags:
  🚨 Requires Immediate Action: {summary['flags']['requires_immediate_action']}
  ⚠️  Requires Attention: {summary['flags']['requires_attention']}
  📅 Overdue: {summary['flags']['overdue']}
  ⏰ Long Overdue (>90 days): {summary['flags']['long_overdue']}
  🚫 High Risk No Job Ref: {summary['flags']['high_risk_no_job']}

Risk Score:
  Average: {summary['risk_score_avg']:.1f}/100
  Maximum: {summary['risk_score_max']}/100

Top Categories:
"""

        # Sort categories by count
        sorted_categories = sorted(
            summary['by_category'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for category, count in sorted_categories[:5]:
            text += f"  - {category.replace('_', ' ').title()}: {count}\n"

        return text
