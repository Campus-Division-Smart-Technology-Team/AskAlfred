#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile NLPIntentClassifier.classify_intent on a sample set.

Usage:
  python tools/profile_intent.py
  python tools/profile_intent.py --loops 50 --output profile.out
"""

from __future__ import annotations

import argparse
import cProfile
import random
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from query_core.intent_classifier import (  # noqa: E402
    INTENT_EXAMPLES,
    NLPIntentClassifier,
)


def _build_samples() -> list[str]:
    samples: list[str] = []
    for examples in INTENT_EXAMPLES.values():
        samples.extend(examples)
    return samples


def _run_profile(loops: int, output_path: Path) -> None:
    classifier = NLPIntentClassifier()
    samples = _build_samples()
    if not samples:
        return
    rng = random.Random(42)

    def _workload() -> None:
        for _ in range(loops):
            q = rng.choice(samples)
            classifier.classify_intent(q)

    profiler = cProfile.Profile()
    profiler.enable()
    _workload()
    profiler.disable()
    profiler.dump_stats(str(output_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile intent classification and write profile.out"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=100,
        help="Number of classify_intent calls to run (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile.out"),
        help="Output path for cProfile stats (default: profile.out)",
    )
    parser.add_argument(
        "--streamlit-ui",
        action="store_true",
        help="Render a minimal Streamlit UI before profiling (for overhead)",
    )
    return parser.parse_args()


def _maybe_run_streamlit_ui() -> None:
    st.set_page_config(page_title="Intent Profiler", layout="wide")
    st.title("Intent Classifier Profiling")
    st.write("Running profiling workload...")
    st.empty()


def main() -> None:
    args = _parse_args()

    if args.streamlit_ui:
        _maybe_run_streamlit_ui()

    _run_profile(args.loops, args.output)
    print(f"Wrote profile stats to {args.output}")


if __name__ == "__main__":
    main()
