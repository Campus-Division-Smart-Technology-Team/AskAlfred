#!/usr/bin/env python3

from __future__ import annotations

from core import clients


def test_get_redis_uses_bounded_timeouts_and_ssl(monkeypatch):
    captured_kwargs = {}

    class FakeRedis:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(clients, "Redis", FakeRedis)
    monkeypatch.setattr(clients.ClientManager, "_redis", None)
    monkeypatch.setenv("REDIS_HOST", "redis.example.test")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_USERNAME", "default")
    monkeypatch.setenv("REDIS_PASSWORD", "secret")
    monkeypatch.setenv("REDIS_DB", "2")
    monkeypatch.setenv("REDIS_SSL", "true")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "1.5")
    monkeypatch.setenv("REDIS_SOCKET_CONNECT_TIMEOUT", "2.5")
    monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "10")

    redis_client = clients.ClientManager.get_redis()

    assert isinstance(redis_client, FakeRedis)
    assert captured_kwargs["host"] == "redis.example.test"
    assert captured_kwargs["port"] == 6380
    assert captured_kwargs["db"] == 2
    assert captured_kwargs["ssl"] is True
    assert captured_kwargs["socket_timeout"] == 1.5
    assert captured_kwargs["socket_connect_timeout"] == 2.5
    assert captured_kwargs["health_check_interval"] == 10
    assert captured_kwargs["retry_on_timeout"] is False
