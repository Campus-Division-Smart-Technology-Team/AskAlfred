"""
Table parsing helpers for FRA action plan extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
import logging
from typing import Optional, Protocol, cast

from config.constant import RISK_LEVEL_MAP

from ..types import ParsedRowData


@dataclass
class _LayoutRowData:
    desc: list[str] = field(default_factory=list)
    solution: list[str] = field(default_factory=list)
    person: list[str] = field(default_factory=list)
    job: list[str] = field(default_factory=list)
    expected: list[str] = field(default_factory=list)
    complete: list[str] = field(default_factory=list)


class _TableParserBase(Protocol):
    verbose: bool
    logger: logging.Logger

    def _find_risk_level(self, content: str) -> Optional[str]:
        ...

    def _normalise_table_text(self, text: str) -> str:
        ...

    def _is_valid_row(self, issue_num: str, content: str) -> tuple[bool, Optional[str]]:
        ...

    def _parse_table_rows_layout(
        self,
        section_text: str,
        item_key: str,
    ) -> tuple[list[dict], list[str]]:
        ...


class _TableParserMixin(_TableParserBase):
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
                for j in range(i + 1, min(i + 5, len(lines))):
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

    def _parse_table_rows(
        self: _TableParserBase,
        section_text: str,
        item_key: str,
    ) -> tuple[list[dict], list[str]]:
        parsed_rows: list[dict] = []
        warnings: list[str] = []

        layout_rows, layout_warnings = self._parse_table_rows_layout(
            section_text,
            item_key,
        )
        if layout_rows:
            def _filter_invalid_layout_rows(rows: list[dict]) -> list[dict]:
                valid_rows: list[dict] = []
                for row in rows:
                    issue_num = row.get("issue_num")
                    row_data = row.get("row_data") or {}
                    content = " ".join(
                        part for part in (
                            row_data.get("issue_description"),
                            row_data.get("proposed_solution"),
                        )
                        if part
                    ).strip()
                    if issue_num and content:
                        is_valid, _ = cast(
                            tuple[bool, Optional[str]],
                            self._is_valid_row(issue_num, content),
                        )
                        if is_valid:
                            valid_rows.append(row)
                return valid_rows

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

            layout_rows = _filter_invalid_layout_rows(layout_rows)
            min_layout_rows = 5
            if len(layout_rows) >= min_layout_rows and _layout_rows_usable(layout_rows):
                return layout_rows, layout_warnings
            warnings.append(
                "Layout parse quality low; falling back to text-based row parsing"
            )
        warnings.extend(layout_warnings)

        section_text = cast(
            str,
            self._normalise_table_text(section_text),
        )
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

            if not self._is_valid_row(issue_num, content)[0]:
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

            parsed_rows.append(
                {
                    "issue_num": issue_num,
                    "risk_level": risk_level,
                    "content": content,
                    "row_data": None,
                }
            )

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

            parsed_rows.append(
                {
                    "issue_num": issue_num,
                    "risk_level": risk_level,
                    "content": combined_text,
                    "row_data": row_data,
                }
            )

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
