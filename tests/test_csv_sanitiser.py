"""Tests for spreadsheet-formula neutralisation in CSV export paths."""

import pytest

from security.csv_sanitiser import csv_safe_cell, neutralise_csv_formula
from tools.extract_index_tocsv import format_value_for_csv


@pytest.mark.parametrize(
    "value",
    ["=cmd|A1", "+SUM(A1:A2)", "-10+20", "@HYPERLINK(...)", "  =1+1"],
)
def test_exported_csv_values_neutralise_formulas(value):
    assert format_value_for_csv(value).startswith("'")
    assert neutralise_csv_formula(value) == "'" + value


@pytest.mark.parametrize("value", ["Senate House", "FRA 2024", "", "a-b"])
def test_benign_values_are_unchanged(value):
    assert neutralise_csv_formula(value) == value


def test_csv_safe_cell_only_touches_strings():
    assert csv_safe_cell(-5) == -5
    assert csv_safe_cell(True) is True
    assert csv_safe_cell(None) is None
    assert csv_safe_cell("=1+1") == "'=1+1"
