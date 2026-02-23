from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from pdfminer.high_level import extract_text


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


def main() -> int:
    # File lives in tests/, so repo root is the parent of this directory.
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "RFM-FRA-GoldneyHall-2026-01.pdf"
    out_path = repo_root / "data" / "RFM-FRA-GoldneyHall-2026-01_riskitem_parsed.txt"

    parser_mod = _load_parser(repo_root)
    text = extract_text(str(pdf_path)) or ""

    parser = parser_mod.FRAActionPlanParser(verbose=False)
    items, confidence = parser.extract_risk_items(
        item_text=text,
        item_key=str(pdf_path),
        canonical_building="Goldney Hall",
        page_texts=None,
    )

    lines = []
    lines.append(f"source: {pdf_path}")
    lines.append(f"items: {len(items)}")
    lines.append(f"confidence: {confidence.overall}")
    lines.append("")

    for idx, item in enumerate(items, start=1):
        lines.append(f"--- item {idx} ---")
        lines.append(f"issue_number: {item.get('issue_number')}")
        lines.append(f"risk_level: {item.get('risk_level')}")
        lines.append(f"issue_description: {item.get('issue_description')}")
        lines.append(f"proposed_solution: {item.get('proposed_solution')}")
        lines.append(f"person_responsible: {item.get('person_responsible')}")
        lines.append(f"job_reference: {item.get('job_reference')}")
        lines.append(
            f"expected_completion_date: {item.get('expected_completion_date')}")
        lines.append(
            f"actual_completion_date: {item.get('actual_completion_date')}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
