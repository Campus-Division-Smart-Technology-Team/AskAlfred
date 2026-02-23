"""
Row parsing helpers for FRA action plan extraction.
"""

from __future__ import annotations

import re

from ..types import ParsedRowData


class _RowParserMixin:
    """
        Mixin for parsing individual action plan rows into structured data.

        Challenges:
        - Inconsistent formatting (multi-line fields, missing delimiters)
        - Extracting person responsible without a clear label
        - Distinguishing expected vs actual completion dates
    """
    # Completion markers in the "Checked as complete" column
    COMPLETION_MARKERS: list[str] = [
        r"complete",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",  # Date format
        r"[A-Z]{2,}\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}",  # Name + date
    ]

    def _parse_row_content(self, content: str) -> ParsedRowData:
        """
        Parse individual row content into field dictionary.

        Expected structure (from sample FRA):
        - Issue description (multi-line)
        - Proposed solution (multi-line)
        - Person responsible (name)
        - Job reference (numeric)
        - Expected completion date
        - Actual completion (name + date OR just date)
        """

        # Split into potential sections
        lines = content.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Initialise fields
        parsed: ParsedRowData = {
            "issue_description": "",
            "proposed_solution": "",
            "person_responsible": None,
            "job_reference": None,
            "expected_completion_date": None,
            "actual_completion_date": None,
            "figure_references": [],
        }

        # Extract figure references
        parsed["figure_references"] = re.findall(
            r"\bFig(?:ure)?\.?\s*(\d+)", content, re.IGNORECASE)

        # Extract issue description (heuristic: split on "which require attention"
        # only when it clearly introduces a follow-on section)
        description_end = re.search(
            r"(?:which require attention)\s*:?\s*",
            content,
            re.IGNORECASE,
        )
        if description_end:
            remainder = content[description_end.end():].strip()
            # Only split if there is meaningful trailing content (likely a new section).
            if len(remainder) >= 20:
                parsed["issue_description"] = content[:description_end.start()
                                                      ].strip()
                parsed["proposed_solution"] = remainder
        # Additional heuristic: split on "were noted;" if it clearly ends the issue description.
        if not parsed["issue_description"]:
            noted_end = re.search(
                r"were noted\s*;\s*", content, re.IGNORECASE)
            if noted_end:
                remainder = content[noted_end.end():].strip()
                if len(remainder) >= 20:
                    parsed["issue_description"] = content[:noted_end.end()
                                                          ].strip()
                    parsed["proposed_solution"] = remainder

        # Extract person responsible (name pattern: FirstName LastName)
        # Look for name immediately before a month/year completion date.
        MONTH_NAMES = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

        person_before_date = re.search(
            rf"(?!{MONTH_NAMES})([A-Z][a-zA-Z''-]+(?:[\s/\n]+[A-Z][a-zA-Z''-]+)+)\s+"
            rf"(?:{MONTH_NAMES})\s+\d{{2,4}}\b",
            content,
        )
        if person_before_date:
            parsed["person_responsible"] = person_before_date.group(
                1).replace("\n", " ").strip()

        # If no job reference, look for a name preceding a month-year completion date.
        if not parsed["person_responsible"]:
            person_before_date = re.search(
                r"([A-Z][a-zA-Z'â€™-]+(?:[\s/\n]+[A-Z][a-zA-Z'â€™-]+)*)\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
                r"January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+\d{2,4}\b",
                content,
            )
            if person_before_date:
                parsed["person_responsible"] = person_before_date.group(
                    1).replace("\n", " ").strip()
        if not parsed["person_responsible"]:
            org_before_date = re.search(
                r"\b([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*)\b\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                content,
            )
            if org_before_date:
                parsed["person_responsible"] = org_before_date.group(
                    1).strip()

        # Extract job reference (6+ digit number)
        job_ref_pattern = r"\b(\d{6,})\b"
        order_match = re.search(
            r"Order\s+number[:\s]+(\d{6,})\b", content, re.IGNORECASE)
        if order_match:
            parsed["job_reference"] = order_match.group(1)
        job_refs = re.findall(job_ref_pattern, content)
        if job_refs:
            # Filter out dates (which also have 6-8 digits)
            job_refs = [
                ref
                for ref in job_refs
                if not re.match(r"\d{2}[/-]\d{2}[/-]\d{4}", ref)
            ]
            if job_refs and not parsed["job_reference"]:
                parsed["job_reference"] = job_refs[0]

        # Extract explicit completion date (avoid mis-assigning to expected date)
        date_token = (
            r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}"
            r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{4})"
        )
        complete_match = re.search(
            rf"Complete\s*[--:]\s*({date_token})",
            content,
            re.IGNORECASE,
        )
        completed_match = re.search(
            rf"Completed\s+({date_token})",
            content,
            re.IGNORECASE,
        )
        if complete_match:
            parsed["actual_completion_date"] = complete_match.group(1)
        if completed_match and not parsed["actual_completion_date"]:
            parsed["actual_completion_date"] = completed_match.group(1)

        # Extract dates
        date_patterns = [
            r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # DD/MM/YYYY
            r"\b(\d{1,2}/\d{1,2}/\d{2})\b",  # DD/MM/YY
            r"\b(\d{1,2}-\d{1,2}-\d{4})\b",  # DD-MM-YYYY
            r"\b(\d{1,2}-\d{1,2}-\d{2})\b",  # DD-MM-YY
            # Month YYYY (full or abbreviated)
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2})\b",
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{2,4})\b",
        ]

        dates_found: list[str] = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(
                pattern, content, re.IGNORECASE))

        def _normalise_two_digit_year(value: str) -> str:
            match = re.search(
                r"\b(\d{1,2}[/-]\d{1,2}[/-])(\d{2})\b", value)
            if match:
                return f"{match.group(1)}20{match.group(2)}"
            match = re.search(
                r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)(\d{2})\b",
                value,
                re.IGNORECASE,
            )
            if match:
                return f"{match.group(1)}20{match.group(2)}"
            return value

        if parsed["actual_completion_date"]:
            parsed["actual_completion_date"] = _normalise_two_digit_year(
                parsed["actual_completion_date"]
            )
        if parsed["expected_completion_date"]:
            parsed["expected_completion_date"] = _normalise_two_digit_year(
                parsed["expected_completion_date"]
            )
        dates_found = [_normalise_two_digit_year(
            d) for d in dates_found]

        # First date is usually expected completion (skip explicit completion date)
        if dates_found and not parsed["expected_completion_date"]:
            if parsed["actual_completion_date"]:
                expected_candidates = [
                    d for d in dates_found
                    if d != parsed["actual_completion_date"]
                ]
            else:
                expected_candidates = dates_found
            if expected_candidates:
                parsed["expected_completion_date"] = expected_candidates[0]

        # Check for completion markers
        if not parsed["actual_completion_date"]:
            for marker_pattern in self.COMPLETION_MARKERS:
                if re.search(marker_pattern, content, re.IGNORECASE):
                    # If we found completion, try to extract date
                    if len(dates_found) > 1:
                        # Last date
                        parsed["actual_completion_date"] = dates_found[-1]
                    break

        # Split content into description and solution
        # Heuristic: solution often starts with action verbs
        action_verbs = [
            "upgrade", "replace", "install", "remove", "repair",
            "ensure", "relocate", "clear", "implement", "review",
            "fire-stop", "secure", "provide", "create", "inspect", "train",
        ]
        # Only fall back to action-verb split if description_end didn't already work
        if not parsed["issue_description"]:
            solution_start_idx = None
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(
                    line_lower.startswith(verb)
                    or re.search(rf"\b{re.escape(verb)}\b", line_lower)
                    for verb in action_verbs
                ):
                    solution_start_idx = i
                    break

            if solution_start_idx is not None:
                parsed["issue_description"] = " ".join(
                    lines[:solution_start_idx])
                parsed["proposed_solution"] = " ".join(
                    lines[solution_start_idx:])
            else:
                parsed["issue_description"] = content
                parsed["proposed_solution"] = ""
        # If description is empty but solution has text, treat it as description.
        if not parsed["issue_description"] and parsed["proposed_solution"]:
            parsed["issue_description"] = parsed["proposed_solution"]
            parsed["proposed_solution"] = ""

        return parsed
