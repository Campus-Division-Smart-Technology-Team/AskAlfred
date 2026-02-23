from __future__ import annotations

import importlib.util
import sys
import types
from datetime import date, timedelta
import pytest
from pathlib import Path


def _load_triage(repo_root: Path):
    # Stub date_utils to keep tests lightweight.
    stub_date_utils = types.ModuleType("date_utils")

    def parse_iso_date(value):
        if not value:
            return None
        text = str(value)
        if "T" in text:
            text = text.split("T", 1)[0]
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    setattr(stub_date_utils, "parse_iso_date", parse_iso_date)
    sys.modules["date_utils"] = stub_date_utils

    # Load config.constant and expose only constants via a stub config module.
    spec_const = importlib.util.spec_from_file_location(
        "config.constant", repo_root / "config" / "constant.py"
    )
    if spec_const is None or spec_const.loader is None:
        raise ImportError("Could not load config.constant")
    mod_const = importlib.util.module_from_spec(spec_const)
    sys.modules["config.constant"] = mod_const
    spec_const.loader.exec_module(mod_const)

    config_pkg = types.ModuleType("config")
    for name, value in vars(mod_const).items():
        if not name.startswith("_"):
            setattr(config_pkg, name, value)
    sys.modules["config"] = config_pkg

    # Ensure FRA_CRITICAL_OVERDUE_DAYS is available as a fallback
    if not hasattr(config_pkg, "FRA_CRITICAL_OVERDUE_DAYS"):
        setattr(config_pkg, "FRA_CRITICAL_OVERDUE_DAYS", 60)

    # Load fra.types, fra.enrichment, and fra.triage without fra/__init__.
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

    spec_enrichment = importlib.util.spec_from_file_location(
        "fra.enrichment", repo_root / "fra" / "enrichment.py"
    )
    if spec_enrichment is None or spec_enrichment.loader is None:
        raise ImportError("Could not load fra.enrichment")
    mod_enrichment = importlib.util.module_from_spec(spec_enrichment)
    mod_enrichment.__package__ = "fra"
    sys.modules["fra.enrichment"] = mod_enrichment
    spec_enrichment.loader.exec_module(mod_enrichment)

    spec_triage = importlib.util.spec_from_file_location(
        "fra.triage", repo_root / "fra" / "triage.py"
    )
    if spec_triage is None or spec_triage.loader is None:
        raise ImportError("Could not load fra.triage")
    mod_triage = importlib.util.module_from_spec(spec_triage)
    mod_triage.__package__ = "fra"
    sys.modules["fra.triage"] = mod_triage
    spec_triage.loader.exec_module(mod_triage)

    return mod_triage, config_pkg


def _base_item(**overrides):
    base = {
        "canonical_building_name": "Test Building",
        "issue_number": "1",
        "risk_level": 3,
        "risk_level_text": None,
        "fra_assessment_date": None,
        "expected_completion_date": None,
        "actual_completion_date": None,
        "job_reference": None,
        "issue_description": "Replace defective fire doors.",
        "proposed_solution": "Replace fire doors in corridor.",
        "risk_item_id": None,
        "document_type": None,
        "namespace": None,
        "ingestion_timestamp": None,
        "is_current": True,
        "superseded_by": None,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def triage_computer():
    repo_root = Path(__file__).resolve().parents[1]
    triage_mod, _ = _load_triage(repo_root)

    class StubEnricher:
        def compute_completion_status(
            self,
            *,
            expected_date,
            completion_date,
            today,
            data_quality_issues=None,
        ):
            if completion_date:
                return "complete"
            if expected_date and expected_date < today:
                return "overdue"
            return "open"

        def enrich_base_fields(self, *, risk_item, risk_level):
            return {
                "risk_item_id": risk_item.get("risk_item_id"),
                "risk_level_text": "Stub",
                "ingestion_timestamp": "1970-01-01T00:00:00Z",
                "document_type": risk_item.get("document_type"),
                "namespace": risk_item.get("namespace"),
                "is_current": risk_item.get("is_current"),
                "superseded_by": risk_item.get("superseded_by"),
            }

    config = triage_mod.TriageConfig(
        long_overdue_days=90,
        critical_overdue_days=180,
        extreme_overdue_days=365,
        max_days_sanity=3650,
    )
    return triage_mod.FRATriageComputer(
        verbose=False,
        config=config,
        enricher=StubEnricher(),
    )


def test_overdue_flags_and_attention_priority(triage_computer):

    today = date.today()
    item = _base_item(
        risk_level=3,
        job_reference="123456",
        fra_assessment_date=(today - timedelta(days=120)).isoformat(),
        expected_completion_date=(today - timedelta(days=10)).isoformat(),
    )

    enriched = triage_computer.compute_flags(item)
    assert enriched["flag_overdue"] is True
    assert enriched["flag_long_overdue"] is False
    assert enriched["requires_attention"] is True
    assert triage_computer.get_priority_label(enriched) == "URGENT"


def test_critical_overdue_triggers_immediate_action(triage_computer):
    repo_root = Path(__file__).resolve().parents[1]
    _, config = _load_triage(repo_root)
    today = date.today()
    overdue_days = int(config.FRA_CRITICAL_OVERDUE_DAYS) + \
        1  # pylint: disable=no-member
    item = _base_item(
        risk_level=2,
        job_reference="123456",
        fra_assessment_date=(today - timedelta(days=400)).isoformat(),
        expected_completion_date=(
            today - timedelta(days=overdue_days)).isoformat(),
    )

    enriched = triage_computer.compute_flags(item)
    assert enriched["flag_critical_overdue"] is True
    assert enriched["requires_immediate_action"] is True
    assert triage_computer.get_priority_label(enriched) == "CRITICAL"


def test_complete_priority_label(triage_computer):
    today = date.today()
    item = _base_item(
        risk_level=5,
        job_reference="123456",
        fra_assessment_date=(today - timedelta(days=30)).isoformat(),
        expected_completion_date=(today - timedelta(days=5)).isoformat(),
        actual_completion_date=(today - timedelta(days=1)).isoformat(),
    )

    enriched = triage_computer.compute_flags(item)
    assert enriched["completion_status"] == "complete"
    assert triage_computer.get_priority_label(enriched) == "COMPLETE"
