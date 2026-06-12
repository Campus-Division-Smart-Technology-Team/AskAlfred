#!/usr/bin/env python3
"""Development-time environment bootstrap utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_BOOTSTRAP_COMPLETE = False
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_LOCAL_ENV_FLAG_NAMES = {"ALLOW_LOCAL_ENV", "ENVIRONMENT"}


def _is_truthy_env_flag(name: str) -> bool:
    """Return True when an environment flag uses a supported truthy value."""
    return os.getenv(name, "").strip().lower() in _TRUTHY_VALUES


def _is_truthy_value(value: object) -> bool:
    """Return True when a value uses a supported truthy representation."""
    return str(value or "").strip().lower() in _TRUTHY_VALUES


def load_local_env() -> None:
    """Optionally load the repository `.env` for local development only."""
    global _ENV_BOOTSTRAP_COMPLETE

    if _ENV_BOOTSTRAP_COMPLETE:
        return

    env_path = Path(__file__).resolve().parent / ".env"
    dotenv_flags: dict[str, object] = {}

    if env_path.exists():
        try:
            from dotenv import dotenv_values, load_dotenv
        except Exception:  # pylint: disable=broad-except
            logger.info(
                "Skipping local .env load: python-dotenv is unavailable in this environment."
            )
            _ENV_BOOTSTRAP_COMPLETE = True
            return

        try:
            dotenv_values_result = dotenv_values(env_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to inspect local .env from %s: %s", env_path, exc)
            _ENV_BOOTSTRAP_COMPLETE = True
            return

        dotenv_flags = {
            key: value
            for key, value in dotenv_values_result.items()
            if key in _LOCAL_ENV_FLAG_NAMES
        }
    else:
        load_dotenv = None

    environment_value = os.getenv("ENVIRONMENT") or dotenv_flags.get("ENVIRONMENT")
    environment = str(environment_value or "development").strip().lower()
    if not environment:
        environment = "development"

    allow_local_env_value = os.getenv("ALLOW_LOCAL_ENV") or dotenv_flags.get(
        "ALLOW_LOCAL_ENV"
    )
    if not _is_truthy_value(allow_local_env_value):
        logger.info("Skipping local .env load: ALLOW_LOCAL_ENV is not enabled.")
        _ENV_BOOTSTRAP_COMPLETE = True
        return

    if environment != "development":
        logger.info(
            "Skipping local .env load: ENVIRONMENT=%s is not local development.",
            environment,
        )
        _ENV_BOOTSTRAP_COMPLETE = True
        return

    if not env_path.exists():
        logger.info("Skipping local .env load: %s does not exist.", env_path)
        _ENV_BOOTSTRAP_COMPLETE = True
        return

    try:
        load_dotenv(env_path, override=False)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load local .env from %s: %s", env_path, exc)
        _ENV_BOOTSTRAP_COMPLETE = True
        return

    logger.info("Loaded local .env from %s with override=False.", env_path)
    _ENV_BOOTSTRAP_COMPLETE = True
