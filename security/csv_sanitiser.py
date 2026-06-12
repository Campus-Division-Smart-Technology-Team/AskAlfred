#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV output sanitisation helpers.

Spreadsheet applications execute cell values beginning with "=", "+", "-",
or "@" as formulas, so any exported metadata could trigger formula injection
when a CSV is opened in Excel/LibreOffice (OWASP CSV injection). Every CSV
export path must route string cell values through these helpers.
"""

from typing import Any

CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")


def neutralise_csv_formula(value: str) -> str:
    """Return a CSV-safe cell value for spreadsheet applications."""
    stripped = value.lstrip()
    if stripped.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def csv_safe_cell(value: Any) -> Any:
    """Neutralise string cells; pass other types (numbers, bools) through."""
    if isinstance(value, str):
        return neutralise_csv_formula(value)
    return value
