#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client initialisation for Pinecone, OpenAI, and Redis.
"""

import os
from typing import Optional
from pinecone import Pinecone
from openai import OpenAI
from redis import Redis

from alfred_exceptions import ConfigError

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
        """
        if cls._pc is not None:
            return cls._pc

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ConfigError(
                "PINECONE_API_KEY is not set. "
                "Set it in the environment or mock get_pc() during tests."
            )

        cls._pc = Pinecone(api_key=api_key)
        return cls._pc

    @classmethod
    def get_oai(cls) -> OpenAI:
        if cls._oai is not None:
            return cls._oai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("Missing OPENAI_API_KEY")

        cls._oai = OpenAI(api_key=api_key)
        return cls._oai

    @classmethod
    def get_redis(cls) -> Redis:
        """
        Lazy-load Redis client.
        Only creates client when first needed.
        """
        if cls._redis is not None:
            return cls._redis
        redis_host = os.environ.get("REDIS_HOST")
        redis_port = int(os.environ.get("REDIS_PORT", 0)
                         ) if os.environ.get("REDIS_PORT") else 0
        redis_username = os.environ.get("REDIS_USERNAME", "")
        redis_password = os.environ.get("REDIS_PASSWORD", "")

        if not redis_host or not redis_port:
            raise ConfigError(
                "REDIS_HOST or REDIS_PORT not set"
            )
        try:
            port = int(redis_port)
        except ValueError as exc:
            raise ConfigError(f"Invalid REDIS_PORT: {redis_port}") from exc

        cls._redis = Redis(
            host=redis_host,
            port=port,
            decode_responses=True,
            username=redis_username,
            password=redis_password,
            health_check_interval=30,
        )
        return cls._redis


def get_pc() -> Pinecone:
    """Lazy-load Pinecone client."""
    return ClientManager.get_pc()


def get_oai() -> OpenAI:
    """Lazy-load OpenAI client."""
    return ClientManager.get_oai()


def get_redis() -> Redis:
    """Lazy-load Redis client."""
    return ClientManager.get_redis()
