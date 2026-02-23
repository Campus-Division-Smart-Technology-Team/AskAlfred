"""
Section parsing helpers for FRA action plan extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Protocol

from config.constant import SPLIT_RISK_FIXES


class _SectionParserBase(Protocol):
    verbose: bool
    logger: logging.Logger


class _SectionParserMixin(_SectionParserBase):

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
