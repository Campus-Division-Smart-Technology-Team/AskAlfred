from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump full-document text from a PDF to a txt file, preserving layout."
    )
    parser.add_argument(
        "--pdf",
        default="data/RFM-FRA-GoldneyHall-2026-01.pdf",
        help="Path to the PDF (repo-relative or absolute).",
    )
    parser.add_argument(
        "--out",
        default="data/RFM-FRA-GoldneyHall-2026-01_full_document.txt",
        help="Output txt path (repo-relative or absolute).",
    )
    return parser


def _resolve_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def main() -> int:
    # File lives in tests/, so repo root is the parent of this directory.
    repo_root = Path(__file__).resolve().parents[1]
    args = _build_parser().parse_args()
    pdf_path = _resolve_path(repo_root, args.pdf)
    out_path = _resolve_path(repo_root, args.out)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract text using pdftotext with -layout flag to preserve spatial formatting.
    # pdftotext is part of poppler-utils: install with `apt install poppler-utils`
    # or `brew install poppler` on macOS. On Windows, install Poppler and ensure
    # `pdftotext` is available on PATH.
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        text = tmp_path.read_text(encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise RuntimeError(f"pdftotext failed:\n{stderr}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    lines = []
    lines.append(f"source: {pdf_path}")
    lines.append(f"characters: {len(text)}")
    lines.append("")
    lines.append(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
