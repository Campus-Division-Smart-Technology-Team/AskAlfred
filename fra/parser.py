"""
FRA Action Plan Parser

Extracts structured risk items from Fire Risk Assessment action plan tables.
Handles multiple table formats and provides confidence scoring.
"""

import re
import logging
from typing import Optional, Any
from dataclasses import dataclass
from date_utils import parse_date_to_iso, parse_iso_date
from pinecone_utils import sanitise_metadata_for_pinecone
from .doc_metadata import extract_assessment_date
from .types import ParsedRowData, RiskItem
from .parse_helpers.parse_section import _SectionParserMixin
from .parse_helpers.parse_table import _TableParserMixin
from .parse_helpers.parse_row import _RowParserMixin


@dataclass
class ParsingConfidence:
    """Track parsing confidence for quality monitoring."""
    overall: float  # 0.0 - 1.0
    field_scores: dict[str, float]
    warnings: list[str]


class FRAActionPlanParser(_SectionParserMixin, _TableParserMixin, _RowParserMixin):
    """
    Extract structured risk items from FRA action plan tables.

    Supports multiple table formats:
    - Standard UoB template
    - Legacy formats
    - Multi-page tables
    """

    def __init__(self, verbose: bool = False):
        """
        Initialise parser.

        Args:
            verbose: Enable detailed logging
        """
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def extract_risk_items(
        self,
        item_text: str,
        item_key: str,
        canonical_building: str,
        page_texts: Optional[list[str]] = None
    ) -> tuple[list[RiskItem], ParsingConfidence]:
        """
        Parse FRA action plan table and return structured records.

        Args:
            item_text: Full text of FRA
            item_key: Document identifier
            canonical_building: Resolved building name
            page_texts: Optional list of per-page text for evidence linking

        Returns:
            Tuple of (risk_items, confidence_report)
        """
        self.logger.info("Extracting risk items from %s", item_key)

        # 1. Extract metadata from header
        assessment_date = self._extract_assessment_date(item_text)
        if not assessment_date:
            self.logger.warning(
                "Could not extract assessment date from %s", item_key)

        # 2. Locate action plan section
        action_plan_text, start_page = self._extract_action_plan_section(
            item_text, page_texts
        )

        if not action_plan_text:
            self.logger.error(
                "No action plan section found in %s", item_key)
            return [], ParsingConfidence(
                overall=0.0,
                field_scores={},
                warnings=["No action plan section found"]
            )

        # 3. Parse table rows
        parsed_rows, parsing_warnings = self._parse_table_rows(
            action_plan_text,
            item_key,
        )

        # Convert parsed rows to risk items
        risk_items = []
        for row in parsed_rows:
            row_data = row.get("row_data")
            if row_data is None:
                row_data = self._parse_row_content(row['content'])
            # Ensure typed shape for downstream usage
            row["row_data"] = row_data
            risk_item = self._build_risk_item(
                issue_num=row['issue_num'],
                risk_level=row['risk_level'],
                row_data=row_data,
                item_key=item_key,
                canonical_building=canonical_building,
                assessment_date=assessment_date,
                page_num=start_page
            )
            risk_items.append(risk_item)

        # 4. Link evidence (figures and pages)
        if page_texts:
            figure_map = self._build_figure_map(page_texts)
            for item in risk_items:
                self._link_evidence(item, figure_map, page_texts)

        # 5. Compute confidence score
        confidence = self._compute_confidence(
            risk_items, parsing_warnings)

        self.logger.info(
            "Extracted %s risk items from %s (confidence: %.2f)",
            len(risk_items),
            item_key,
            confidence.overall,
        )

        return risk_items, confidence

    def extract_assessment_date(self, text: str) -> Optional[str]:
        """Public wrapper to extract assessment date from document text."""
        return self._extract_assessment_date(text)

    def _extract_assessment_date(self, text: str) -> Optional[str]:
        """Extract assessment date from document header."""
        return extract_assessment_date(text)

    def _build_risk_item(
        self,
        issue_num: str,
        risk_level: str,
        row_data: ParsedRowData,
        item_key: str,
        canonical_building: str,
        assessment_date: Optional[str],
        page_num: int
    ) -> RiskItem:
        """Convert parsed row data into structured risk item record."""

        risk_level_int = int(
            risk_level) if risk_level is not None else 0
        assessment_date_int = None
        if assessment_date:
            parsed_date = parse_iso_date(assessment_date)
            if parsed_date:
                assessment_date_int = (
                    parsed_date.year * 10000
                    + parsed_date.month * 100
                    + parsed_date.day
                )

        # Computed/enriched fields are set in triage to keep parser extractive.
        completion_status = None
        risk_level_text = None
        risk_item_id = None
        ingestion_timestamp = None
        document_type = None
        namespace = None
        is_current = None
        superseded_by = None

        return {
            # Identity
            "risk_item_id": risk_item_id,
            "partition_key": f"{canonical_building}#{issue_num}",
            "canonical_building_name": canonical_building,
            "fra_document_key": item_key,
            "fra_assessment_date": assessment_date,
            "fra_assessment_date_int": assessment_date_int,
            "issue_number": issue_num,

            # Risk assessment
            "issue_description": row_data["issue_description"].strip(),
            "risk_level": risk_level_int,
            "risk_level_text": risk_level_text,

            # Remediation
            "proposed_solution": row_data["proposed_solution"].strip(),
            "person_responsible": row_data["person_responsible"],
            "job_reference": row_data["job_reference"],

            # Timeline
            "expected_completion_date": parse_date_to_iso(row_data["expected_completion_date"]),
            "actual_completion_date": parse_date_to_iso(row_data["actual_completion_date"]),
            "completion_status": completion_status,

            # Evidence
            "figure_references": row_data["figure_references"],
            "page_references": [],  # Populated by _link_evidence
            "action_plan_page": page_num,

            # Metadata
            "document_type": document_type,
            "namespace": namespace,
            "ingestion_timestamp": ingestion_timestamp,

            # Triage flags (computed later)
            "is_current": is_current,
            "superseded_by": superseded_by,
        }

    def _build_figure_map(self, page_texts: list[str]) -> dict[str, int]:
        """Build map of figure numbers to page numbers."""
        figure_map: dict[str, int] = {}

        for page_num, page_text in enumerate(page_texts, start=1):
            # Find all "Figure X" references
            figures = re.findall(
                r"Figure\s+(\d+)", page_text, re.IGNORECASE)

            for fig_num in figures:
                if fig_num not in figure_map:
                    figure_map[fig_num] = page_num

        return figure_map

    def _link_evidence(
        self,
        risk_item: RiskItem,
        figure_map: dict[str, int],
        page_texts: list[str]
    ) -> None:
        """Link risk item to evidence (figures and pages)."""

        # Add page numbers for referenced figures
        page_refs: set[int] = set()
        for fig_num in risk_item["figure_references"]:
            if fig_num in figure_map:
                page_refs.add(figure_map[fig_num])

        # Also search for issue description in pages
        issue_keywords = risk_item["issue_description"][:100].lower(
        ).split()
        # Only substantial words
        issue_keywords = [w for w in issue_keywords if len(w) > 4]

        for page_num, page_text in enumerate(page_texts, start=1):
            page_text_lower = page_text.lower()
            keyword_matches = sum(
                1 for kw in issue_keywords if kw in page_text_lower)

            # If many keywords match, this page likely discusses the issue
            if keyword_matches >= 3:
                page_refs.add(page_num)

        risk_item["page_references"] = sorted(page_refs)

    def _compute_confidence(
        self,
        risk_items: list[RiskItem],
        warnings: list[str]
    ) -> ParsingConfidence:
        """Compute parsing confidence score."""

        if not risk_items:
            return ParsingConfidence(
                overall=0.0,
                field_scores={},
                warnings=warnings + ["No risk items extracted"]
            )

        # Score each field across all items
        field_scores: dict[str, float] = {}

        fields_to_check = [
            "issue_description",
            "proposed_solution",
            "person_responsible",
            "job_reference",
            "expected_completion_date"
        ]

        for field in fields_to_check:
            non_empty = sum(
                1 for item in risk_items if item.get(field))
            field_scores[field] = non_empty / len(risk_items)

        # Overall score is weighted average
        weights = {
            "issue_description": 0.3,
            "proposed_solution": 0.2,
            "person_responsible": 0.2,
            "job_reference": 0.15,
            "expected_completion_date": 0.15
        }

        overall = sum(
            field_scores[field] * weights[field]
            for field in fields_to_check
        )

        # Penalise for warnings
        warning_penalty = min(len(warnings) * 0.05, 0.2)
        overall = max(0.0, overall - warning_penalty)

        return ParsingConfidence(
            overall=overall,
            field_scores=field_scores,
            warnings=warnings
        )


def sanitise_risk_item_for_metadata(risk_item: RiskItem) -> dict[str, Any]:
    """
    Return a metadata-safe copy of a risk item with None values removed.

    Pinecone metadata does not allow nulls, and MetadataValidator rejects them.
    None values are encoded with the Pinecone null sentinel and restored on read.
    """
    return sanitise_metadata_for_pinecone(dict(risk_item))


def parse_action_plan_in_process(
    item_text: str,
    item_key: str,
    canonical_building: str,
    verbose: bool = False,
) -> tuple[list[RiskItem], ParsingConfidence]:
    """
    Top-level worker for ProcessPoolExecutor.

    Keep this function module-level so it is picklable on Windows.
    """
    parser = FRAActionPlanParser(verbose=verbose)
    return parser.extract_risk_items(
        item_text=item_text,
        item_key=item_key,
        canonical_building=canonical_building,
        page_texts=None,
    )
