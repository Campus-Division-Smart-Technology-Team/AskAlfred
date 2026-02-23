from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_fra_parser(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    parser_path = repo_root / "fra" / "parser.py"
    if not parser_path.exists():
        raise FileNotFoundError(f"Parser module not found: {parser_path}")
    if "fra" not in sys.modules:
        fra_pkg = types.ModuleType("fra")
        fra_pkg.__path__ = [str(repo_root / "fra")]
        sys.modules["fra"] = fra_pkg
    spec = importlib.util.spec_from_file_location("fra.parser", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {parser_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fra.parser"] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse FRA action plan items from full-document text."
    )
    parser.add_argument(
        "--text",
        default="data/RFM-FRA-GoldneyHall-2026-01.txt",
        help="Path to the full-document text file.",
    )
    parser.add_argument(
        "--item-key",
        default="RFM-FRA-GoldneyHall-2026-01",
        help="Document identifier used in parsed items.",
    )
    parser.add_argument(
        "--building",
        default="Goldney Hall",
        help="Canonical building name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of items to print (0 prints all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable parser debug logging.",
    )
    return parser


def _resolve_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = _build_parser().parse_args()

    text_path = _resolve_path(repo_root, args.text)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    text = text_path.read_text(encoding="utf-8", errors="replace")
    fra_parser = _load_fra_parser(repo_root)
    parser = fra_parser.FRAActionPlanParser(verbose=args.verbose)
    items, confidence = parser.extract_risk_items(
        text,
        args.item_key,
        args.building,
    )

    limit = args.limit
    if limit == 0:
        sample = items
    else:
        sample = items[:limit]

    print(f"items: {len(items)}")
    print(f"confidence: {confidence.overall:.2f}")
    print("warnings:", confidence.warnings)
    print("sample:")
    print(json.dumps(sample, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
