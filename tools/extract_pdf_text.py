"""
Extract raw text from PDF files and save as .txt for review.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.document_content import extract_text


def _build_output_path(pdf_path: Path, out: str | None) -> Path:
    if out:
        out_path = Path(out)
        if out_path.is_dir():
            return out_path / f"{pdf_path.stem}.txt"
        return out_path
    return pdf_path.with_suffix(".txt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract raw text from a PDF and save to a .txt file."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--out",
        help="Output file path or directory (defaults to the same location as the PDF)",
    )
    parser.add_argument(
        "--layout",
        action="store_true",
        help="Use pdftotext --layout for layout-preserving text extraction.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit("Input must be a .pdf file")

    out_path = _build_output_path(pdf_path, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.layout:
        pdftotext = shutil.which("pdftotext")
        if not pdftotext:
            raise SystemExit("pdftotext not found on PATH")
        cmd = [
            pdftotext,
            "-layout",
            "-enc",
            "UTF-8",
            str(pdf_path),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise SystemExit(f"pdftotext failed ({result.returncode}): {err}")
        text = out_path.read_text(encoding="utf-8", errors="replace")
    else:
        data = pdf_path.read_bytes()
        text = extract_text(pdf_path.name, data)
        out_path.write_text(text, encoding="utf-8")

    print(f"Wrote {len(text)} chars to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
