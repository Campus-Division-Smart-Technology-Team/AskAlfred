"""
CLI package exports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_batch_ingest import main, parse_args

__all__ = ["main", "parse_args"]
