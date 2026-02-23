from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types


def _load_fra_parser():
    repo_root = Path(__file__).resolve().parents[1]
    parser_path = repo_root / "fra" / "parser.py"
    types_path = repo_root / "fra" / "types.py"

    if "fra" not in sys.modules:
        pkg = types.ModuleType("fra")
        pkg.__path__ = [str(repo_root / "fra")]
        sys.modules["fra"] = pkg

    types_spec = spec_from_file_location("fra.types", types_path)
    if types_spec is None or types_spec.loader is None:
        raise ImportError("Could not load fra.types module")
    types_module = module_from_spec(types_spec)
    sys.modules["fra.types"] = types_module
    types_spec.loader.exec_module(types_module)

    parser_spec = spec_from_file_location("fra.parser", parser_path)
    if parser_spec is None or parser_spec.loader is None:
        raise ImportError("Could not load fra.parser module")
    parser_module = module_from_spec(parser_spec)
    parser_module.__package__ = "fra"
    sys.modules["fra.parser"] = parser_module
    parser_spec.loader.exec_module(parser_module)
    return parser_module


def test_extract_risk_item_inline_risk_level_and_issue_number_delimiter():
    text = """FIRE RISK ASSESSMENT ACTION PLAN
Where similar issues present (such as faults with multiple fire doors or breaches of compartmentalisation), these should be l isted as one action but with all locations
identified.  Note that whilst individual issues may be low risk (e.g. simple fault with a single fire door), if accumulated (simple faults with multiple fire doors) it may be
appropriate to raise the risk level.  Equally, a low-level risk may escalate if left unattended from one review to the next.
Issue  Risk
Level
Issue description and location Proposed solution Person
responsible
Job
reference
number
Expected
completion
(date)
Checked as
complete
(names & date)
1
Tolerable During discussions with the Business
school, it has been identified there
are not enough Fire Wardens trained
based in the building, this could
mean that evacuations will not be
orchestrated effectively and could
lead to risk of life.
The school will be asked to train
more fire wardens to ensure
enough wardens based in the
building.
School office  14/08/2024
2
Moderate There is only one level egress route
from the ground floor, this could lead
to issues for any users with
accessibility issues if
"""

    fra_parser = _load_fra_parser()
    parser = fra_parser.FRAActionPlanParser(verbose=False)
    items, confidence = parser.extract_risk_items(
        item_text=text,
        item_key="example_fra.pdf",
        canonical_building="Example Building",
        page_texts=None,
    )

    assert len(items) == 2

    first = items[0]
    assert first["issue_number"] == "1"
    assert first["risk_level"] == 2
    assert "During discussions with the Business" in first["issue_description"]
    assert "train more fire wardens" in first["proposed_solution"]
    assert first["person_responsible"] == "School office"
    assert first["expected_completion_date"] == "2024-08-14"
    assert first["actual_completion_date"] is None

    second = items[1]
    assert second["issue_number"] == "2"
    assert second["risk_level"] == 3
    assert "There is only one level egress route" in second["issue_description"]
