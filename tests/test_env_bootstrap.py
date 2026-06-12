#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from core import env_bootstrap


def _reset_bootstrap_state(monkeypatch):
    monkeypatch.setattr(env_bootstrap, "_ENV_BOOTSTRAP_COMPLETE", False)


def _install_fake_dotenv(monkeypatch, values: dict[str, str]):
    def fake_dotenv_values(_path):
        return values

    def fake_load_dotenv(_path, override=False):
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
        return True

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(dotenv_values=fake_dotenv_values, load_dotenv=fake_load_dotenv),
    )


def _install_fake_path(monkeypatch, *, env_exists: bool):
    class FakeEnvPath:
        def __init__(self, path: str):
            self.path = path

        def exists(self):
            return env_exists

        def __str__(self):
            return self.path

    class FakeRepoPath:
        def __init__(self, path: str):
            self.path = path

        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def __truediv__(self, other: str):
            return FakeEnvPath(f"{self.path}\\{other}")

    monkeypatch.setattr(
        env_bootstrap,
        "Path",
        lambda *_args, **_kwargs: FakeRepoPath(r"C:\fake-repo"),
    )


def test_load_local_env_skips_when_not_opted_in(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.delenv("ALLOW_LOCAL_ENV", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("BOOTSTRAP_TEST_VALUE", raising=False)
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(monkeypatch, {"BOOTSTRAP_TEST_VALUE": "from-dotenv"})

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") is None


def test_load_local_env_loads_when_opted_in_for_development(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setenv("ALLOW_LOCAL_ENV", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("BOOTSTRAP_TEST_VALUE", raising=False)
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(monkeypatch, {"BOOTSTRAP_TEST_VALUE": "from-dotenv"})

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") == "from-dotenv"


def test_load_local_env_loads_when_opted_in_from_dotenv(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.delenv("ALLOW_LOCAL_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("BOOTSTRAP_TEST_VALUE", raising=False)
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(
        monkeypatch,
        {
            "ALLOW_LOCAL_ENV": "true",
            "ENVIRONMENT": "development",
            "BOOTSTRAP_TEST_VALUE": "from-dotenv",
        },
    )

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") == "from-dotenv"


def test_load_local_env_skips_in_production_even_when_opted_in(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setenv("ALLOW_LOCAL_ENV", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("BOOTSTRAP_TEST_VALUE", raising=False)
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(monkeypatch, {"BOOTSTRAP_TEST_VALUE": "from-dotenv"})

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") is None


def test_load_local_env_does_not_override_existing_environment(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setenv("ALLOW_LOCAL_ENV", "yes")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("BOOTSTRAP_TEST_VALUE", "real-env")
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(monkeypatch, {"BOOTSTRAP_TEST_VALUE": "from-dotenv"})

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") == "real-env"


def test_load_local_env_skips_unknown_environment(monkeypatch):
    _reset_bootstrap_state(monkeypatch)
    monkeypatch.setenv("ALLOW_LOCAL_ENV", "on")
    monkeypatch.setenv("ENVIRONMENT", "qa")
    monkeypatch.delenv("BOOTSTRAP_TEST_VALUE", raising=False)
    _install_fake_path(monkeypatch, env_exists=True)
    _install_fake_dotenv(monkeypatch, {"BOOTSTRAP_TEST_VALUE": "from-dotenv"})

    env_bootstrap.load_local_env()

    assert os.environ.get("BOOTSTRAP_TEST_VALUE") is None
