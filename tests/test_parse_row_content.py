"""Tests for the _parse_row_content method of FRAActionPlanParser, focusing on date extraction logic."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_parser(repo_root: Path):
    # Stub date_utils to avoid heavy imports
    stub_date_utils = types.ModuleType("date_utils")

    def parse_date_to_iso(value):
        return value

    setattr(stub_date_utils, "parse_date_to_iso", parse_date_to_iso)
    sys.modules["date_utils"] = stub_date_utils

    # Stub config package and load constant.py directly (avoids streamlit deps)
    config_pkg = types.ModuleType("config")
    config_pkg.__path__ = [str(repo_root / "config")]
    sys.modules["config"] = config_pkg

    spec_const = importlib.util.spec_from_file_location(
        "config.constant", repo_root / "config" / "constant.py"
    )
    if spec_const is None or spec_const.loader is None:
        raise ImportError("Could not load config.constant")
    mod_const = importlib.util.module_from_spec(spec_const)
    sys.modules["config.constant"] = mod_const
    spec_const.loader.exec_module(mod_const)

    # Stub emojis module
    stub_emojis = types.ModuleType("emojis")
    setattr(stub_emojis, "EMOJI_TICK", "✓")
    sys.modules["emojis"] = stub_emojis

    # Load fra.parser without fra/__init__
    fra_pkg = types.ModuleType("fra")
    fra_pkg.__path__ = [str(repo_root / "fra")]
    sys.modules["fra"] = fra_pkg

    spec_types = importlib.util.spec_from_file_location(
        "fra.types", repo_root / "fra" / "types.py"
    )
    if spec_types is None or spec_types.loader is None:
        raise ImportError("Could not load fra.types")
    mod_types = importlib.util.module_from_spec(spec_types)
    sys.modules["fra.types"] = mod_types
    spec_types.loader.exec_module(mod_types)

    spec_parser = importlib.util.spec_from_file_location(
        "fra.parser", repo_root / "fra" / "parser.py"
    )
    if spec_parser is None or spec_parser.loader is None:
        raise ImportError("Could not load fra.parser")
    mod_parser = importlib.util.module_from_spec(spec_parser)
    mod_parser.__package__ = "fra"
    sys.modules["fra.parser"] = mod_parser
    spec_parser.loader.exec_module(mod_parser)

    return mod_parser


def _get_parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser_mod = _load_parser(repo_root)
    return parser_mod.FRAActionPlanParser(verbose=False)


def test_completed_sets_actual_date():
    parser = _get_parser()
    content = (
        "Replace fire doors in corridor. "
        "Expected completion date: 12/05/2024. "
        "Completed May 2024."
    )
    parsed = parser._parse_row_content(content)
    assert parsed.get("expected_completion_date") == "12/05/2024"
    assert parsed.get("actual_completion_date") == "May 2024"


def test_complete_marker_sets_actual_date():
    parser = _get_parser()
    content = (
        "Upgrade fire alarm panel. "
        "Expected completion date: 01/06/2024. "
        "Complete - 15/06/2024."
    )
    parsed = parser._parse_row_content(content)
    assert parsed.get("expected_completion_date") == "01/06/2024"
    assert parsed.get("actual_completion_date") == "15/06/2024"


def test_completion_marker_fallback_uses_last_date():
    parser = _get_parser()
    content = (
        "Install fire-stop around service penetrations. "
        "Expected completion date: 01/06/2024. "
        "Checked as complete 15/06/2024."
    )
    parsed = parser._parse_row_content(content)
    assert parsed.get("expected_completion_date") == "01/06/2024"
    assert parsed.get("actual_completion_date") == "15/06/2024"


def test_person_before_date_does_not_set_job_reference():
    parser = _get_parser()
    content = "Jane Doe May 2024 - install signage in stairwells."
    parsed = parser._parse_row_content(content)
    assert parsed.get("person_responsible") == "Jane Doe"
    assert parsed.get("job_reference") is None
