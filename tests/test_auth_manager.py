#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest

import auth_manager
from alfred_exceptions import ConfigError
from auth_manager import (
    _build_auth_context_from_claims,
    _get_or_create_auth_flow,
    _normalise_query_params,
)


def test_normalise_query_params_flattens_lists():
    params = {
        "code": ["abc", "latest"],
        "state": "xyz",
        "empty": [],
    }

    assert _normalise_query_params(params) == {
        "code": "latest",
        "state": "xyz",
    }


def test_build_auth_context_from_claims_prefers_preferred_username_and_roles():
    claims = {
        "preferred_username": "user@bristol.ac.uk",
        "name": "Test User",
        "tid": "tenant-123",
        "roles": ["viewer", "exporter"],
    }

    auth_context = _build_auth_context_from_claims(claims)

    assert auth_context.user_id == "user@bristol.ac.uk"
    assert auth_context.display_name == "Test User"
    assert auth_context.tenant_id == "tenant-123"
    assert auth_context.roles == ("viewer", "exporter")
    assert auth_context.authenticated is True
    assert auth_context.auth_source == "entra_id"


def test_build_auth_context_from_claims_falls_back_to_object_id():
    claims = {
        "oid": "object-id-123",
        "tid": "tenant-123",
    }

    auth_context = _build_auth_context_from_claims(claims)

    assert auth_context.user_id == "object-id-123"
    assert auth_context.display_name == "object-id-123"
    assert auth_context.email is None


def test_get_or_create_auth_flow_requires_client_secret(monkeypatch):
    monkeypatch.setattr(
        auth_manager,
        "st",
        SimpleNamespace(session_state={}),
    )
    monkeypatch.setattr(
        auth_manager.SecureCredentialManager,
        "get_missing_azure_credentials",
        classmethod(lambda cls, include_client_secret=True: ["AZURE_CLIENT_SECRET"]),
    )

    with pytest.raises(ConfigError, match="AZURE_CLIENT_SECRET"):
        _get_or_create_auth_flow()


def test_get_or_create_auth_flow_builds_flow_when_secret_present(monkeypatch):
    fake_streamlit = SimpleNamespace(session_state={})
    fake_flow = {
        "auth_uri": "https://login.microsoftonline.com/example/oauth2/v2.0/authorize"
    }

    class FakeMsalApp:
        def initiate_auth_code_flow(self, scopes, redirect_uri, response_mode):
            assert scopes == ["email", "User.Read"]
            assert redirect_uri == auth_manager.AUTH_REDIRECT_URI
            assert response_mode == "query"
            return fake_flow

    monkeypatch.setattr(auth_manager, "st", fake_streamlit)
    monkeypatch.setattr(
        auth_manager.SecureCredentialManager,
        "get_missing_azure_credentials",
        classmethod(lambda cls, include_client_secret=True: []),
    )
    monkeypatch.setattr(auth_manager, "build_msal_app", lambda **kwargs: FakeMsalApp())
    monkeypatch.setattr(
        auth_manager,
        "get_login_scopes",
        lambda: ["email", "User.Read"],
    )

    flow = _get_or_create_auth_flow()

    assert flow == fake_flow
    assert fake_streamlit.session_state[auth_manager.AUTH_FLOW_SESSION_KEY] == fake_flow
