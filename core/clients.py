#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client initialisation for Pinecone, OpenAI, and Redis.

Credentials are loaded on-demand via SecureCredentialManager:
- Not stored in memory longer than necessary
- Fresh fetch from environment each time
- Sanitized error messages to prevent credential exposure
"""

import logging
import os
from typing import Optional

from openai import OpenAI
from pinecone import Pinecone
from redis import Redis

from auth.credential_manager import SecureCredentialManager
from core.alfred_exceptions import ConfigError
from security.log_sanitiser import sanitise_message

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_REDIS_SOCKET_TIMEOUT = 5.0
_DEFAULT_REDIS_CONNECT_TIMEOUT = 5.0
_DEFAULT_REDIS_HEALTH_CHECK_INTERVAL = 30.0


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY_VALUES


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0")
    return value


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0")
    return value

# print("pinecone loaded from:", Pinecone.__module__)
# print("openai loaded from:", OpenAI.__module__)
# print("redis loaded from:", Redis.__module__)


class ClientManager:
    """Manages lazy-loaded clients for Pinecone, OpenAI, and Redis."""

    _pc: Optional[Pinecone] = None
    _oai: Optional[OpenAI] = None
    _redis: Optional[Redis] = None

    @classmethod
    def get_pc(cls) -> Pinecone:
        """
        Lazy-load Pinecone client.
        Only creates client when first needed.
        Credentials fetched fresh from environment each time.
        """
        if cls._pc is not None:
            return cls._pc

        try:
            # Get credential fresh from environment (not cached in memory)
            api_key = SecureCredentialManager.get_pinecone_api_key()
            cls._pc = Pinecone(api_key=api_key)
            return cls._pc
        except KeyError as e:
            logging.error("Pinecone API key not configured")
            raise ConfigError(
                "PINECONE_API_KEY not set. Please configure credentials."
            ) from e
        except Exception as e:
            # Log sanitized error - credential manager has already redacted the key
            logging.error("Failed to initialise Pinecone: %s", sanitise_message(str(e)))
            raise ConfigError("Failed to initialise Pinecone client") from e

    @classmethod
    def get_oai(cls) -> OpenAI:
        """
        Lazy-load OpenAI client.
        Credentials fetched fresh from environment each time.
        """
        if cls._oai is not None:
            return cls._oai

        try:
            # Get credential fresh from environment (not cached in memory)
            api_key = SecureCredentialManager.get_openai_api_key()
            cls._oai = OpenAI(api_key=api_key)
            return cls._oai
        except KeyError as e:
            logging.error("OpenAI API key not configured")
            raise ConfigError(
                "OPENAI_API_KEY not set. Please configure credentials."
            ) from e
        except Exception as e:
            # Log sanitized error - credential manager has already redacted the key
            logging.error("Failed to initialise OpenAI: %s", sanitise_message(str(e)))
            raise ConfigError("Failed to initialise OpenAI client") from e

    @classmethod
    def get_redis(cls) -> Redis:
        """
        Lazy-load Redis client.
        Only creates client when first needed.
        """
        if cls._redis is not None:
            return cls._redis

        redis_host = os.environ.get("REDIS_HOST")
        redis_port_str = os.environ.get("REDIS_PORT", "0")
        redis_username = os.environ.get("REDIS_USERNAME", "")
        redis_password = os.environ.get("REDIS_PASSWORD", "")
        redis_db = _get_int_env("REDIS_DB", 0)
        socket_timeout = _get_float_env(
            "REDIS_SOCKET_TIMEOUT", _DEFAULT_REDIS_SOCKET_TIMEOUT
        )
        socket_connect_timeout = _get_float_env(
            "REDIS_SOCKET_CONNECT_TIMEOUT", _DEFAULT_REDIS_CONNECT_TIMEOUT
        )
        health_check_interval = _get_float_env(
            "REDIS_HEALTH_CHECK_INTERVAL", _DEFAULT_REDIS_HEALTH_CHECK_INTERVAL
        )

        if not redis_host:
            raise ConfigError("REDIS_HOST is not set in environment")

        try:
            redis_port = int(redis_port_str)
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"Invalid REDIS_PORT: {redis_port_str}") from exc

        if redis_port <= 0 or redis_port > 65535:
            raise ConfigError("REDIS_PORT must be between 1 and 65535")

        try:
            cls._redis = Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                username=redis_username if redis_username else None,
                password=redis_password if redis_password else None,
                ssl=_is_truthy_env("REDIS_SSL"),
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                retry_on_timeout=False,
            )
            return cls._redis
        except Exception as e:
            # Log sanitized error
            logging.error("Failed to initialise Redis: %s", sanitise_message(str(e)))
            raise ConfigError("Failed to initialise Redis client") from e


def get_pc() -> Pinecone:
    """Lazy-load Pinecone client."""
    return ClientManager.get_pc()


def get_oai() -> OpenAI:
    """Lazy-load OpenAI client."""
    return ClientManager.get_oai()


def get_redis() -> Redis:
    """Lazy-load Redis client."""
    return ClientManager.get_redis()
