#!/usr/bin/env python3
"""Development-time environment bootstrap utilities."""

from __future__ import annotations

from pathlib import Path

_ENV_LOADED = False


def load_local_env() -> None:
    """Best-effort load of the repository .env without overriding real env vars."""
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except Exception:  # pylint: disable=broad-except
        return

    _ENV_LOADED = True
