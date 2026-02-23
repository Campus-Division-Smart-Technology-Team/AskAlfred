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
from config.constant import SPLIT_RISK_FIXES, RISK_LEVEL_MAP
from .types import ParsedRowData, RiskItem


@dataclass
class ParsingConfidence:
    """Track parsing confidence for quality monitoring."""
    overall: float  # 0.0 - 1.0
    field_scores: dict[str, float]
    warnings: list[str]


@dataclass
class _LayoutRowData:
    desc: list[str]
    solution: list[str]
    person: list[str]
    job: list[str]
    expected: list[str]
    complete: list[str]


class FRAActionPlanParser:
    """
    Extract structured risk items from FRA action plan tables.

    Supports multiple table formats:
    - Standard UoB template
    - Legacy formats
    - Multi-page tables
    """

    # Completion markers in the "Checked as complete" column
    COMPLETION_MARKERS: list[str] = [
        r"complete",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",  # Date format
        r"[A-Z]{2,}\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}",  # Name + date
    ]

    def __init__(self, verbose: bool = False):
        """
        Initialise parser.

        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def _extract_action_plan_section(
        self,
        text: str,
        page_texts: Optional[list[str]] = None
    ) -> tuple[Optional[str], int]:
        """
        Locate the action plan table section.

        Returns:
            Tuple of (section_text, start_page_number)
        """
        # Markers for action plan start
        start_patterns = [
            r"FIRE RISK ASSESSMENT ACTION PLAN",
            r"ACTION\s+PLAN",
            r"Issue\s+Risk\s+Issue\s+description",
            r"Issue\s+Risk\s+Issue\s+description\s+and\s+location",
            r"Issue\s+Ris(?:k)?\s*Lev(?:el)?\s+Issue\s+description",
        ]

        # Try to find in full text first
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if self.verbose:
                    self.logger.debug(
                        "Found action plan with pattern: %s", pattern)
                    self.logger.debug("Position: %s", match.start())

                section_text = text[match.start():]

                # Find page number if we have page-level text
                start_page = 1
                if page_texts:
                    char_count = 0
                    for page_num, page_text in enumerate(page_texts, start=1):
                        char_count += len(page_text)
                        if char_count >= match.start():
                            start_page = page_num
                            break

                # Stop at next major section (if any)
                end_patterns = [
                    r"\n\s*Photographs\s*\n",
                    r"\n\s*Appendix",
                    r"\n\s*EVALUATION OF A FIRE",
                ]

                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, section_text)
                    if end_match:
                        if self.verbose:
                            self.logger.debug(
                                "Found end marker: %s", end_pattern)
                            self.logger.debug(
                                "Section length: %s chars", end_match.start())
                        section_text = section_text[:end_match.start()]
                        break

                return section_text, start_page

        if self.verbose:
            self.logger.error("No action plan section found with any pattern")

        return None, 0

    def _normalise_table_text(self, text: str) -> str:
        """
        Normalise PDF-extracted table text for more reliable row parsing.

        This targets common extraction artifacts such as broken lines, header noise,
        and isolated numeric cells that get separated from their content.
        """
        for pattern, repl in SPLIT_RISK_FIXES:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        lines = text.splitlines()
        cleaned: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                cleaned.append("")
                i += 1
                continue

            # Drop common footer/header noise
            if re.match(r"^HAS-FT-\d+\s+Version:", line, re.IGNORECASE):
                i += 1
                continue
            if re.match(r"^Page\s+\d+\s+of\s+\d+", line, re.IGNORECASE):
                i += 1
                continue

            # Join hyphen-wrapped lines
            if line.endswith("-") and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                line = f"{line[:-1]}{next_line}"
                i += 1

            cleaned.append(line)
            i += 1

        # Collapse excessive blank lines
        normalised_lines: list[str] = []
        blank_run = 0
        for line in cleaned:
            if not line:
                blank_run += 1
                if blank_run <= 2:
                    normalised_lines.append(line)
                continue
            blank_run = 0
            normalised_lines.append(line)

        return "\n".join(normalised_lines).strip()

    def _is_valid_row(self, issue_num: str, content: str) -> tuple[bool, Optional[str]]:
        """
        Validate that this looks like a real action plan row.

        Returns:
            (is_valid, reason_if_invalid)
        """
        try:
            issue_int = int(issue_num)
        except ValueError:
            return False, "Issue number is not an integer"

        # Issue numbers should be reasonable (1-100 typically)
        if issue_int > 100:
            return False, f"Issue number {issue_int} too large (likely a date)"

        # Content should be substantial (not just a date or short phrase)
        content_stripped = content.strip()
        if len(content_stripped) < 30:
            return False, f"Content too short ({len(content_stripped)} chars)"

        # Should not start with common non-row patterns
        non_row_patterns = [
            r"^\s*Complete:",
            r"^\s*Fig(?:ure)?\s+\d+",
            r"^\s*Page\s+\d+",
            r"^\s*HAS-FT-\d+",
        ]

        for pattern in non_row_patterns:
            if re.match(pattern, content_stripped, re.IGNORECASE):
                return False, f"Matches non-row pattern: {pattern}"

        # Should not be primarily just dates
        date_matches = re.findall(
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', content_stripped)
        if date_matches and len(' '.join(date_matches)) > len(content_stripped) * 0.7:
            return False, "Content is primarily dates"

        return True, None

    def _find_risk_level(self, content: str) -> Optional[str]:
        """
        Find risk level in content with multiple strategies.

        Returns:
            Risk level (1-5) or None
        """
        lines = content.splitlines()

        # Strategy 1: Look for dedicated risk level line
        for line in lines[:10]:  # Check first 10 lines
            # Pattern: optional "Risk Level:" followed by a digit
            line_match = re.match(
                r"^\s*(?:Risk\s*Level[:\s]*)?([1-5])\s*$",
                line,
                re.IGNORECASE,
            )
            if line_match:
                return line_match.group(1)

        # Strategy 2: Look for single digit on its own line (common in PDFs)
        for line in lines[:10]:
            if re.match(r"^\s*([1-5])\s*$", line):
                return line.strip()

        # Strategy 3: Handle split text like "Ris\nk\nLev\nel\n3"
        # Look for isolated digit after potential header fragments
        for i, line in enumerate(lines[:15]):
            if re.search(r"(?:Ris|Lev|el)", line, re.IGNORECASE):
                # Check next few lines for a digit
                for j in range(i+1, min(i+5, len(lines))):
                    if re.match(r"^\s*([1-5])\s*$", lines[j]):
                        return lines[j].strip()

        # Strategy 4: Look for any risk level keywords in the first 15 lines
        risk_label_map = {v.lower(): k for k, v in RISK_LEVEL_MAP.items()}
        for line in lines[:15]:
            token = line.strip().lower()
            token = re.sub(r"[^a-z]", "", token)  # keep only letters
            if token in risk_label_map:
                return str(risk_label_map[token])

        # Strategy 5: Inline search (last resort)
        inline_match = re.search(
            r"\b(?:Risk\s*Level[:\s]*)?([1-5])\b",
            content,
            re.IGNORECASE,
        )
        if inline_match:
            return inline_match.group(1)

        inline = re.search(r"\b(trivial|tolerable|moderate|substantial|intolerable)\b",
                           content, re.IGNORECASE)
        if inline:
            return str(risk_label_map.get(inline.group(1).lower()))

        return None

    def _parse_table_rows(self, section_text, item_key) -> tuple[list[dict], list[str]]:
        parsed_rows: list[dict] = []
        warnings: list[str] = []

        layout_rows, layout_warnings = self._parse_table_rows_layout(
            section_text,
            item_key,
        )
        if layout_rows:
            def _layout_rows_usable(rows: list[dict]) -> bool:
                missing_risk = 0
                empty_desc = 0
                for row in rows:
                    row_data = row.get("row_data") or {}
                    risk_level = row.get("risk_level")
                    if risk_level is None:
                        missing_risk += 1
                    issue_desc = (row_data.get(
                        "issue_description") or "").strip()
                    solution = (row_data.get(
                        "proposed_solution") or "").strip()
                    if not issue_desc and not solution:
                        empty_desc += 1
                total = max(len(rows), 1)
                missing_risk_ratio = missing_risk / total
                empty_desc_ratio = empty_desc / total
                return missing_risk_ratio <= 0.5 and empty_desc_ratio <= 0.5

            if _layout_rows_usable(layout_rows):
                return layout_rows, layout_warnings
            warnings.append(
                "Layout parse quality low; falling back to text-based row parsing"
            )
        warnings.extend(layout_warnings)

        section_text = self._normalise_table_text(section_text)
        # Find all lines that are solely an issue number (1–99)
        # These mark the START of each row
        boundary_pattern = re.compile(
            r"(?m)^\s*(?P<num>\d{1,2})"
            r"(?:\s+(?P<rest>(?:[A-Za-z].*|[1-5]\s+[A-Za-z].*)))?\s*$"
        )
        boundaries = []
        for m in boundary_pattern.finditer(section_text):
            num = m.group("num")
            try:
                if int(num) <= 0:
                    continue
            except ValueError:
                continue
            boundaries.append((m.start(), num, m.group("rest")))

        if not boundaries:
            warnings.append(
                "No row boundaries found (no standalone issue numbers)")
            return [], warnings

        for i, (start, issue_num, first_rest) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + \
                1 < len(boundaries) else len(section_text)
            row_text = section_text[start:end].strip()
            lines = row_text.splitlines()
            # If the issue number shares a line with content, keep that content.
            if first_rest:
                content_lines = [first_rest.strip()] + lines[1:]
            else:
                # row_text begins with the issue number line; skip it
                content_lines = lines[1:]
            content = "\n".join(content_lines).strip()

            # Strip figure/diagram noise from the row content
            content = re.sub(
                r"^\s*(?:Fig(?:ure)?\.?\s*\d+|To\s+Fig(?:ure)?\s*\d+)\s*$",
                "",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            content = re.sub(r"\s{2,}", " ", content).strip()

            is_valid, reason = self._is_valid_row(issue_num, content)
            if not is_valid:
                continue

            # If content starts with a risk label, strip it to keep description clean.
            risk_level = None
            leading_match = re.match(
                r"^\s*(?:(?P<num>[1-5])\s+|(?P<label>trivial|tolerable|moderate|substantial|intolerable)\s+)(?P<rest>.+)$",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            if leading_match:
                risk_level = leading_match.group("num")
                if not risk_level and leading_match.group("label"):
                    label = leading_match.group("label").lower()
                    risk_label_map = {
                        v.lower(): k for k, v in RISK_LEVEL_MAP.items()}
                    risk_level = risk_label_map.get(label)
                content = leading_match.group("rest").strip()

            if risk_level is None:
                risk_level = self._find_risk_level(content)
            if risk_level is None:
                warnings.append(f"Row {issue_num}: no risk level found")

            parsed_rows.append({
                'issue_num': issue_num,
                'risk_level': risk_level,
                'content': content,
            })

        return parsed_rows, warnings

    def _parse_table_rows_layout(
        self,
        section_text: str,
        item_key: str,
    ) -> tuple[list[dict], list[str]]:
        """
        Parse action plan rows using fixed column layout (from pdftotext -layout).
        """
        warnings: list[str] = []
        lines = section_text.splitlines()

        header_idx = None
        header_line = None
        for idx, line in enumerate(lines):
            if (
                "Issue description" in line
                and "Proposed solution" in line
                and "Person" in line
                and "Expected" in line
                and "Checked" in line
            ):
                header_idx = idx
                header_line = line
                break

        if header_idx is None or header_line is None:
            return [], []

        def _is_data_row(line: str) -> bool:
            return bool(re.match(r"^\s*\d+\s+[1-5]\s+\S", line))

        def _collect_header_lines() -> list[str]:
            header_lines = [header_line]
            for next_line in lines[header_idx + 1: header_idx + 5]:
                if _is_data_row(next_line):
                    break
                stripped = next_line.strip()
                if not stripped:
                    continue
                if re.search(
                    r"Issue|Ris|Risk|Proposed|solution|Person|Job|Expected|Checked|complete|description|Level",
                    stripped,
                    re.IGNORECASE,
                ):
                    header_lines.append(next_line)
                    continue
                if re.search(r"[A-Za-z]", stripped) and not re.search(r"\d", stripped):
                    header_lines.append(next_line)
            return header_lines

        def _merge_header_lines(header_lines: list[str]) -> str:
            max_len = max(len(line) for line in header_lines)
            merged_chars: list[str] = []
            for idx in range(max_len):
                ch = " "
                for line in header_lines:
                    if idx < len(line) and line[idx] != " ":
                        ch = line[idx]
                        break
                merged_chars.append(ch)
            return "".join(merged_chars).rstrip()

        header_lines = _collect_header_lines()
        merged_header_line = _merge_header_lines(header_lines)

        def _derive_ranges_from_sample(sample_lines: list[str]) -> Optional[dict[str, tuple[int, Optional[int]]]]:
            for line in sample_lines:
                if not _is_data_row(line):
                    continue
                first_non_space = re.search(r"\S", line)
                if not first_non_space:
                    continue
                starts: list[int] = [first_non_space.start()]
                for match in re.finditer(r"\s{4,}", line):
                    next_start = match.end()
                    next_non_space = re.search(r"\S", line[next_start:])
                    if not next_non_space:
                        continue
                    starts.append(next_start + next_non_space.start())
                # Remove duplicates while preserving order.
                unique_starts: list[int] = []
                for value in starts:
                    if value not in unique_starts:
                        unique_starts.append(value)
                if len(unique_starts) < 8:
                    continue
                unique_starts = unique_starts[:8]
                return {
                    "issue": (unique_starts[0], unique_starts[1]),
                    "risk": (unique_starts[1], unique_starts[2]),
                    "desc": (unique_starts[2], unique_starts[3]),
                    "solution": (unique_starts[3], unique_starts[4]),
                    "person": (unique_starts[4], unique_starts[5]),
                    "job": (unique_starts[5], unique_starts[6]),
                    "expected": (unique_starts[6], unique_starts[7]),
                    "complete": (unique_starts[7], None),
                }
            return None

        sample_ranges = _derive_ranges_from_sample(
            lines[header_idx + 1: header_idx + 12])
        if sample_ranges:
            col_ranges = sample_ranges
        else:
            def _find_header_pos(pattern: str, line: str, start: int = 0) -> int:
                match = re.search(pattern, line[start:], re.IGNORECASE)
                if not match:
                    return -1
                return start + match.start()

            issue_start = _find_header_pos(r"\bIssue\b", merged_header_line, 0)
            risk_start = _find_header_pos(
                r"\bRis\b|\bRisk\b", merged_header_line, issue_start + 1)
            desc_start = _find_header_pos(
                r"Issue\s+description", merged_header_line, 0)
            solution_start = _find_header_pos(
                r"Proposed\s+solution", merged_header_line, 0)
            person_start = _find_header_pos(
                r"\bPerson\b", merged_header_line, 0)
            job_start = _find_header_pos(r"\bJob\b", merged_header_line, 0)
            expected_start = _find_header_pos(
                r"\bExpected\b", merged_header_line, 0)
            complete_start = _find_header_pos(
                r"Checked\s+as\s+complete", merged_header_line, 0)

            def _adjust_start(label_start: int, min_start: int) -> int:
                if label_start < 0:
                    return label_start
                if min_start < 0:
                    min_start = 0
                if label_start <= min_start:
                    return label_start
                last_gap_end = None
                for match in re.finditer(r"\s{4,}", merged_header_line[min_start:label_start]):
                    last_gap_end = min_start + match.end()
                if last_gap_end is not None and last_gap_end < label_start:
                    return last_gap_end
                return label_start

            adjusted_issue_start = issue_start
            adjusted_risk_start = _adjust_start(
                risk_start, adjusted_issue_start + 1)
            adjusted_desc_start = _adjust_start(
                desc_start, adjusted_risk_start + 1)
            adjusted_solution_start = _adjust_start(
                solution_start, adjusted_desc_start + 1)
            adjusted_person_start = _adjust_start(
                person_start, adjusted_solution_start + 1)
            adjusted_job_start = _adjust_start(
                job_start, adjusted_person_start + 1)
            adjusted_expected_start = _adjust_start(
                expected_start, adjusted_job_start + 1)
            adjusted_complete_start = _adjust_start(
                complete_start, adjusted_expected_start + 1)

            original_starts = [
                issue_start,
                risk_start,
                desc_start,
                solution_start,
                person_start,
                job_start,
                expected_start,
                complete_start,
            ]
            starts = [
                adjusted_issue_start,
                adjusted_risk_start,
                adjusted_desc_start,
                adjusted_solution_start,
                adjusted_person_start,
                adjusted_job_start,
                adjusted_expected_start,
                adjusted_complete_start,
            ]
            if any(pos < 0 for pos in starts) or not all(
                a < b for a, b in zip(starts, starts[1:])
            ):
                if any(pos < 0 for pos in original_starts) or not all(
                    a < b for a, b in zip(original_starts, original_starts[1:])
                ):
                    warnings.append(
                        "Could not detect ordered action plan column positions"
                    )
                    return [], warnings
                col_ranges = {
                    "issue": (issue_start, risk_start),
                    "risk": (risk_start, desc_start),
                    "desc": (desc_start, solution_start),
                    "solution": (solution_start, person_start),
                    "person": (person_start, job_start),
                    "job": (job_start, expected_start),
                    "expected": (expected_start, complete_start),
                    "complete": (complete_start, None),
                }
            else:
                col_ranges = {
                    "issue": (adjusted_issue_start, adjusted_risk_start),
                    "risk": (adjusted_risk_start, adjusted_desc_start),
                    "desc": (adjusted_desc_start, adjusted_solution_start),
                    "solution": (adjusted_solution_start, adjusted_person_start),
                    "person": (adjusted_person_start, adjusted_job_start),
                    "job": (adjusted_job_start, adjusted_expected_start),
                    "expected": (adjusted_expected_start, adjusted_complete_start),
                    "complete": (adjusted_complete_start, None),
                }

        # col_ranges set above (sample-based or header-based)
        if self.verbose:
            self.logger.debug(
                "Layout column ranges (%s): %s",
                "sample" if sample_ranges else "header",
                col_ranges,
            )

        max_len = max(
            len(line) for line in lines[header_idx + 1:]) if lines[header_idx + 1:] else 0

        def _slice(
            line: str,
            start: Optional[int],
            end: Optional[int],
            col_name: Optional[str] = None,
            min_start: int = 0,
            max_len: int = 0,
        ) -> str:
            """Slice a fixed-width table line with tolerance for pdftotext -layout drift.

            Some rows have cell content starting a few characters left of the
            nominal column boundary (common in fixed-layout PDF extraction).
            We allow small adjustments *within whitespace gaps* between columns
            and also trim accidental carry-over from the next column near the
            right edge of the slice.
            """
            if start is None:
                start = 0
            if len(line) < max_len:
                line = line.ljust(max_len)

            adj_start = max(start, min_start)
            adj_end = end

            apply_start_tolerance = col_name in {
                "desc",
                "solution",
                "person",
                "job",
                "expected",
                "complete",
            }
            apply_end_tolerance = col_name in {
                "desc",
                "solution",
                "person",
                "job",
                "expected",
            }

            # ---- Start tolerance (shift left) ----
            if apply_start_tolerance and start > 0:
                lookback = 18
                window_start = max(min_start, start - lookback)

                # If we are mid-word, rewind to word start.
                if start < len(line) and line[start - 1].isalnum():
                    k = start - 1
                    while k > window_start and line[k - 1].isalnum():
                        k -= 1
                    if k >= min_start:
                        adj_start = min(adj_start, k)

                # Align to the last clear multi-space gap before the start.
                gap_matches = list(
                    re.finditer(r"\s{2,}(?=\S)", line[window_start:start])
                )
                if gap_matches:
                    last = gap_matches[-1]
                    candidate = window_start + last.end()
                    if candidate >= min_start and candidate < start:
                        adj_start = min(adj_start, candidate)

            # ---- End tolerance (trim next-column bleed) ----
            if apply_end_tolerance and adj_end is not None and adj_end > adj_start:
                tail_lookback = 22
                slice_text = line[adj_start:adj_end]
                tail_start = max(0, len(slice_text) - tail_lookback)
                tail = slice_text[tail_start:]
                # If the tail contains a new token preceded by 2+ spaces, cut before it.
                m = list(re.finditer(r"\s{2,}(?=\S)", tail))
                if m:
                    last = m[-1]
                    cut_at = adj_start + tail_start + last.start()
                    if cut_at > adj_start:
                        adj_end = cut_at

            return line[adj_start:adj_end].rstrip()

        parsed_rows: list[dict] = []
        current: Optional[_LayoutRowData] = None
        issue_num = None
        risk_level = None

        def _finalise_row():
            nonlocal current, issue_num, risk_level
            if current is None or not issue_num:
                current = None
                issue_num = None
                risk_level = None
                return

            def _join(parts: list[str]) -> str:
                return re.sub(r"\s+", " ", " ".join(p for p in parts if p).strip())

            current_data = current

            issue_description = _join(current_data.desc)
            proposed_solution = _join(current_data.solution)
            person_responsible = _join(current_data.person) or None
            if person_responsible and re.search(r"\d", person_responsible):
                digit_match = re.search(r"\d{5,}", person_responsible)
                if digit_match:
                    person_responsible = person_responsible[:digit_match.start(
                    )].rstrip()
                    if not person_responsible:
                        person_responsible = None
            job_reference_raw = _join(current_data.job)
            job_reference = None
            if job_reference_raw and job_reference_raw.lower() != "n/a":
                job_match = re.search(r"\b\d{6,}\b", job_reference_raw)
                if job_match:
                    job_reference = job_match.group(0)

            expected_raw = _join(current_data.expected)
            expected_completion_date = expected_raw or None

            complete_raw = _join(current_data.complete)
            date_token = (
                r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}"
                r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
                r"|\d{1,2}[/-]\d{1,2}[/-]\d{4})"
            )
            complete_dates = re.findall(
                date_token, complete_raw, re.IGNORECASE)
            actual_completion_date = complete_dates[-1] if complete_dates else None

            combined_text = " ".join(
                [issue_description, proposed_solution, complete_raw]
            )
            figure_references = re.findall(
                r"\bFig(?:ure)?\.?\s*(\d+)", combined_text, re.IGNORECASE
            )

            row_data: ParsedRowData = {
                "issue_description": issue_description,
                "proposed_solution": proposed_solution,
                "person_responsible": person_responsible,
                "job_reference": job_reference,
                "expected_completion_date": expected_completion_date,
                "actual_completion_date": actual_completion_date,
                "figure_references": figure_references,
            }

            parsed_rows.append({
                "issue_num": issue_num,
                "risk_level": risk_level,
                "content": combined_text,
                "row_data": row_data,
            })

            current = None
            issue_num = None
            risk_level = None

        for line in lines[header_idx + 1:]:
            if "\f" in line:
                line = line.replace("\f", " ")
            if not line.strip():
                continue
            if re.match(r"^\s*HAS-FT-\d+\s+Version:", line, re.IGNORECASE):
                continue
            if re.match(r"^\s*Page\s+\d+\s+of\s+\d+", line, re.IGNORECASE):
                continue
            if re.match(r"^\s*(k|Lev|el|number|n|\(date\))\s*$", line, re.IGNORECASE):
                continue

            # Drop multi-line header continuation rows (common in pdftotext -layout output)
            stripped = line.strip()
            if (not _is_data_row(line)) and re.search(
                r"\b(responsible|reference|completio|names\s*&\s*date|issue\s+description|proposed\s+solution)\b",
                stripped,
                re.IGNORECASE,
            ):
                continue

            normalise_line = line.replace("\f", " ")
            segments = [s.strip()
                        for s in re.split(r"\s{4,}", normalise_line) if s.strip()]
            issue_cell = _slice(
                normalise_line, *col_ranges["issue"], col_name="issue", min_start=0, max_len=max_len).strip()
            risk_cell = _slice(
                normalise_line, *col_ranges["risk"], col_name="risk", min_start=col_ranges["issue"][0], max_len=max_len).strip()

            new_issue = False
            row_match = re.match(
                r"^\s*(\d{1,2})\s{2,}([1-5])\s{2,}\S", normalise_line)
            if row_match:
                issue_candidate = int(row_match.group(1))
                if 1 <= issue_candidate <= 100:
                    new_issue = True
                    issue_cell = row_match.group(1)
                    risk_cell = row_match.group(2)
            if not new_issue and len(segments) >= 2 and segments[0].isdigit() and segments[1].isdigit():
                try:
                    issue_candidate = int(segments[0])
                except ValueError:
                    issue_candidate = 0
                if 1 <= issue_candidate <= 100:
                    new_issue = True
                    issue_cell = segments[0]
                    risk_cell = segments[1]
            if not new_issue and issue_cell.isdigit():
                try:
                    issue_candidate = int(issue_cell)
                except ValueError:
                    issue_candidate = 0
                if 1 <= issue_candidate <= 100:
                    new_issue = True

            if new_issue:
                _finalise_row()
                issue_num = issue_cell
                risk_level = risk_cell if risk_cell.isdigit() else None
                current = _LayoutRowData(
                    desc=[],
                    solution=[],
                    person=[],
                    job=[],
                    expected=[],
                    complete=[],
                )

                if current is None:
                    continue

                def _strip_fig_tokens(value: str) -> str:
                    cleaned = re.sub(
                        r"\b(?:To\s+)?Fig(?:ure)?\.?\s*\d+\b",
                        "",
                        value,
                        flags=re.IGNORECASE,
                    )
                    cleaned = re.sub(r"\s{2,}", " ", cleaned)
                    return cleaned.strip()

                desc_cell = _slice(
                    normalise_line, *col_ranges["desc"], col_name="desc", min_start=col_ranges["risk"][0], max_len=max_len).strip()
                solution_cell = _slice(
                    normalise_line, *col_ranges["solution"], col_name="solution", min_start=col_ranges["desc"][0], max_len=max_len).strip()
                if desc_cell:
                    desc_cell = _strip_fig_tokens(desc_cell)
                if solution_cell:
                    solution_cell = _strip_fig_tokens(solution_cell)
                person_cell = _slice(
                    normalise_line, *col_ranges["person"], col_name="person", min_start=col_ranges["solution"][0], max_len=max_len).strip()
                job_cell = _slice(normalise_line, *col_ranges["job"], col_name="job",
                                  min_start=col_ranges["person"][0], max_len=max_len).strip()
                expected_cell = _slice(
                    normalise_line, *col_ranges["expected"], col_name="expected", min_start=col_ranges["job"][0], max_len=max_len).strip()
                complete_cell = _slice(
                    normalise_line, *col_ranges["complete"], col_name="complete", min_start=col_ranges["expected"][0], max_len=max_len).strip()

                if issue_cell and not new_issue and not issue_cell.isdigit():
                    current.desc.append(issue_cell)
                if desc_cell:
                    current.desc.append(desc_cell)
                if solution_cell:
                    current.solution.append(solution_cell)
                if person_cell:
                    current.person.append(person_cell)
                if job_cell:
                    current.job.append(job_cell)
                if expected_cell:
                    current.expected.append(expected_cell)
                if complete_cell:
                    current.complete.append(complete_cell)

                if risk_level is None and risk_cell.isdigit():
                    risk_level = risk_cell

                # Do NOT finalise here; allow multi-line accumulation.
                continue

            if current is None:
                continue

            # Continuation row: append cells to current
            desc_cell = _slice(
                normalise_line, *col_ranges["desc"], col_name="desc", min_start=col_ranges["risk"][0], max_len=max_len).strip()
            solution_cell = _slice(
                normalise_line, *col_ranges["solution"], col_name="solution", min_start=col_ranges["desc"][0], max_len=max_len).strip()
            person_cell = _slice(
                normalise_line, *col_ranges["person"], col_name="person", min_start=col_ranges["solution"][0], max_len=max_len).strip()
            job_cell = _slice(normalise_line, *col_ranges["job"], col_name="job",
                              min_start=col_ranges["person"][0], max_len=max_len).strip()
            expected_cell = _slice(
                normalise_line, *col_ranges["expected"], col_name="expected", min_start=col_ranges["job"][0], max_len=max_len).strip()
            complete_cell = _slice(
                normalise_line, *col_ranges["complete"], col_name="complete", min_start=col_ranges["expected"][0], max_len=max_len).strip()

            if desc_cell:
                current.desc.append(desc_cell)
            if solution_cell:
                current.solution.append(solution_cell)
            if person_cell:
                current.person.append(person_cell)
            if job_cell:
                current.job.append(job_cell)
            if expected_cell:
                current.expected.append(expected_cell)
            if complete_cell:
                current.complete.append(complete_cell)

        _finalise_row()

        if not parsed_rows:
            warnings.append("No action plan rows parsed from layout text")

        return parsed_rows, warnings

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
        patterns = [
            r"Date of fire risk assessment:\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
            r"Date of fire risk assessment:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
            r"Assessment date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                return parse_date_to_iso(date_str)

        return None

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
            rf"Complete\s*[-–:]\s*({date_token})",
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
            "partition_key": canonical_building,
            "risk_item_id": risk_item_id,
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
    """
    return {key: ("" if value is None else value) for key, value in risk_item.items()}


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
