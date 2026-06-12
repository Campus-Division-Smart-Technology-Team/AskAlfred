#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from auth.credential_manager import SecureCredentialManager
from auth.msal_auth import _get_authority, get_login_scopes
from core.alfred_exceptions import ConfigError


def test_get_authority_uses_configured_tenant(monkeypatch):
    monkeypatch.setattr(
        SecureCredentialManager,
        "get_azure_tenant_id",
        classmethod(lambda cls: "tenant-123"),
    )

    assert _get_authority() == "https://login.microsoftonline.com/tenant-123"


def test_get_authority_requires_tenant_when_fallback_disabled(monkeypatch):
    def _raise_missing_tenant(cls):
        raise KeyError("AZURE_TENANT_ID")

    monkeypatch.setattr(
        SecureCredentialManager,
        "get_azure_tenant_id",
        classmethod(_raise_missing_tenant),
    )

    with pytest.raises(ConfigError):
        _get_authority()


def test_get_authority_uses_common_when_fallback_enabled(monkeypatch):
    def _raise_missing_tenant(cls):
        raise KeyError("AZURE_TENANT_ID")

    monkeypatch.setattr(
        SecureCredentialManager,
        "get_azure_tenant_id",
        classmethod(_raise_missing_tenant),
    )

    assert _get_authority(allow_common_fallback=True) == (
        "https://login.microsoftonline.com/common"
    )


def test_get_login_scopes_uses_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("AUTH_SCOPES", raising=False)

    assert get_login_scopes() == ["email", "User.Read"]


def test_get_login_scopes_parses_comma_separated_values(monkeypatch):
    monkeypatch.setenv("AUTH_SCOPES", "openid, profile, User.Read, Mail.Read")

    assert get_login_scopes() == ["User.Read", "Mail.Read"]


def test_get_login_scopes_filters_reserved_only_values(monkeypatch):
    monkeypatch.setenv("AUTH_SCOPES", "openid,profile,offline_access")

    assert get_login_scopes() == ["email", "User.Read"]


def test_get_login_scopes_logs_when_reserved_scopes_removed(monkeypatch, caplog):
    monkeypatch.setenv("AUTH_SCOPES", "openid, profile, User.Read")

    with caplog.at_level("INFO"):
        scopes = get_login_scopes()

    assert scopes == ["User.Read"]
    assert "reserved MSAL scopes" in caplog.text
    assert "openid, profile" in caplog.text
