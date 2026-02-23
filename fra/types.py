"""
Shared TypedDict contracts for FRA parsing and triage.
"""

from typing import Optional, Literal, TypedDict


CompletionStatus = Literal["open", "overdue", "complete"]


class ParsedRowData(TypedDict):
    """Fields extracted from a raw action-plan row."""

    issue_description: str
    proposed_solution: str
    person_responsible: Optional[str]
    job_reference: Optional[str]
    expected_completion_date: Optional[str]
    actual_completion_date: Optional[str]
    figure_references: list[str]


class RiskItem(TypedDict):
    """Canonical structured FRA risk item produced by the parser."""

    risk_item_id: Optional[str]
    canonical_building_name: str
    fra_document_key: str
    fra_assessment_date: Optional[str]
    fra_assessment_date_int: Optional[int]
    issue_number: str
    issue_description: str
    risk_level: int
    risk_level_text: Optional[str]
    proposed_solution: str
    person_responsible: Optional[str]
    job_reference: Optional[str]
    expected_completion_date: Optional[str]
    actual_completion_date: Optional[str]
    completion_status: Optional[CompletionStatus]
    figure_references: list[str]
    page_references: list[int]
    action_plan_page: int
    document_type: Optional[str]
    namespace: Optional[str]
    ingestion_timestamp: Optional[str]
    is_current: Optional[bool]
    superseded_by: Optional[str]
    partition_key: Optional[str]


class EnrichedRiskItem(RiskItem, total=False):
    """Risk item with triage-computed metrics and flags."""

    days_since_assessment: Optional[int]
    days_until_due: Optional[int]
    days_overdue: Optional[int]
    flag_overdue: bool
    flag_high_risk_no_job: bool
    flag_intolerable: bool
    flag_long_overdue: bool
    flag_critical_overdue: bool
    overdue: bool
    high_risk_no_job: bool
    intolerable: bool
    long_overdue: bool
    critical_overdue: bool
    requires_immediate_action: bool
    requires_attention: bool
    risk_category: str
    risk_score: int
    flag_data_quality_issue: bool
    data_quality_issues: list[str]


class FraVectorExtractResult(TypedDict):
    added: int
    parsing_confidence: Optional[float]
    parsing_warnings: Optional[list[str]]
    parsing_field_scores: Optional[dict[str, float]]
    missing_action_plan: Optional[bool]
    fra_assessment_date: Optional[str]
    fra_assessment_date_int: Optional[int]
