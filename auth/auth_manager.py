#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authentication manager for Microsoft Entra ID sign-in."""

from __future__ import annotations

import html
import logging
import threading
import time
import warnings
from typing import Any

import streamlit as st

from auth.auth_context import ANONYMOUS_AUTH_CONTEXT, AuthContext
from auth.credential_manager import SecureCredentialManager
from auth.msal_auth import build_msal_app, get_login_scopes
from config import (
    ALLOW_ANONYMOUS_DEV,
    AUTH_REDIRECT_URI,
    AUTH_STRICT_TENANT,
    REQUIRE_AUTH,
)
from core.alfred_exceptions import ConfigError
from security.log_sanitiser import sanitise_error

logger = logging.getLogger(__name__)

AUTH_CONTEXT_SESSION_KEY = "auth_context"
AUTH_FLOW_SESSION_KEY = "auth_code_flow"
AUTH_ERROR_SESSION_KEY = "auth_error"
AUTH_FLOW_CACHE_TTL_SECONDS = 900
MICROSOFT_SIGN_IN_BUTTON_MAX_WIDTH_PX = 360
MICROSOFT_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" '
    'width="16" height="16" '
    'aria-hidden="true" focusable="false">'
    '<rect x="1" y="1" width="8" height="8" fill="#f25022"/>'
    '<rect x="11" y="1" width="8" height="8" fill="#7fba00"/>'
    '<rect x="1" y="11" width="8" height="8" fill="#00a4ef"/>'
    '<rect x="11" y="11" width="8" height="8" fill="#ffb900"/>'
    "</svg>"
)


# Streamlit serves sessions from multiple threads, so all access to the
# process-wide auth flow cache must hold this lock.
_AUTH_FLOW_CACHE_LOCK = threading.Lock()


@st.cache_resource
def _get_auth_flow_cache() -> dict[str, tuple[float, dict[str, Any]]]:
    """Return process-wide auth flow cache keyed by OAuth state."""
    return {}


def _prune_auth_flow_cache(cache: dict[str, tuple[float, dict[str, Any]]]) -> None:
    """Drop expired auth flows from the process-wide cache (lock held by caller)."""
    now = time.time()
    expired_states = [
        state
        for state, (created_at, _) in cache.items()
        if now - created_at > AUTH_FLOW_CACHE_TTL_SECONDS
    ]
    for state in expired_states:
        cache.pop(state, None)


def _cache_auth_flow(flow: dict[str, Any]) -> None:
    """Cache auth flow by OAuth state to survive session resets on redirect."""
    state = str(flow.get("state") or "").strip()
    if not state:
        return

    cache = _get_auth_flow_cache()
    with _AUTH_FLOW_CACHE_LOCK:
        _prune_auth_flow_cache(cache)
        cache[state] = (time.time(), flow)


def _get_cached_auth_flow(state: str | None) -> dict[str, Any] | None:
    """Return cached auth flow for a callback state when available."""
    if not state:
        return None

    cache = _get_auth_flow_cache()
    with _AUTH_FLOW_CACHE_LOCK:
        _prune_auth_flow_cache(cache)
        cached_entry = cache.get(state)

    if not cached_entry:
        return None

    _, flow = cached_entry
    return flow if isinstance(flow, dict) else None


def _remove_cached_auth_flow(state: str | None) -> None:
    """Delete cached auth flow for a callback state."""
    if not state:
        return
    with _AUTH_FLOW_CACHE_LOCK:
        _get_auth_flow_cache().pop(state, None)


def authentication_required() -> bool:
    """Return True when the current environment must block anonymous access."""
    return REQUIRE_AUTH or not ALLOW_ANONYMOUS_DEV


def _normalise_query_params(params: dict[str, Any]) -> dict[str, str]:
    """Convert Streamlit query parameters into a flat string dict."""
    normalised: dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, list):
            if value:
                normalised[key] = str(value[-1])
            continue
        normalised[key] = str(value)
    return normalised


def _get_query_params() -> dict[str, str]:
    """Return current query parameters in a Streamlit-version-safe format."""
    if hasattr(st, "query_params"):
        return _normalise_query_params(dict(st.query_params))

    experimental_get_query_params = getattr(st, "experimental_get_query_params", None)
    if callable(experimental_get_query_params):
        params = experimental_get_query_params()
        if isinstance(params, dict):
            return _normalise_query_params(params)

    return {}


def _clear_auth_query_params() -> None:
    """Clear auth-related callback parameters from the URL."""
    if hasattr(st, "query_params"):
        for key in ("code", "state", "session_state", "error", "error_description"):
            if key in st.query_params:
                del st.query_params[key]
        return

    experimental_set_query_params = getattr(st, "experimental_set_query_params", None)
    if callable(experimental_set_query_params):
        experimental_set_query_params()


def _store_auth_context(auth_context: AuthContext) -> AuthContext:
    """Persist the auth context into Streamlit session state."""
    st.session_state[AUTH_CONTEXT_SESSION_KEY] = {
        "user_id": auth_context.user_id,
        "display_name": auth_context.display_name,
        "tenant_id": auth_context.tenant_id,
        "email": auth_context.email,
        "roles": list(auth_context.roles),
        "authenticated": auth_context.authenticated,
        "auth_source": auth_context.auth_source,
    }
    st.session_state.user_id = auth_context.user_id
    st.session_state.user_name = auth_context.display_name
    st.session_state.tenant_id = auth_context.tenant_id
    st.session_state.user_roles = list(auth_context.roles)
    st.session_state.authenticated = auth_context.authenticated
    return auth_context


def _clear_auth_context() -> None:
    """Remove auth-related state from the current session."""
    for key in (
        AUTH_CONTEXT_SESSION_KEY,
        AUTH_FLOW_SESSION_KEY,
        AUTH_ERROR_SESSION_KEY,
        "user_id",
        "user_name",
        "tenant_id",
        "user_roles",
        "authenticated",
    ):
        st.session_state.pop(key, None)


def get_auth_context() -> AuthContext:
    """Return the current auth context from session state or anonymous fallback."""
    raw_context = st.session_state.get(AUTH_CONTEXT_SESSION_KEY)
    if not isinstance(raw_context, dict):
        return ANONYMOUS_AUTH_CONTEXT

    return AuthContext(
        user_id=str(raw_context.get("user_id") or "anonymous"),
        display_name=str(raw_context.get("display_name") or "Anonymous"),
        tenant_id=raw_context.get("tenant_id"),
        email=raw_context.get("email"),
        roles=tuple(raw_context.get("roles") or ()),
        authenticated=bool(raw_context.get("authenticated", False)),
        auth_source=str(raw_context.get("auth_source") or "anonymous"),
    )


def _build_auth_context_from_claims(claims: dict[str, Any]) -> AuthContext:
    """Create an AuthContext from Entra ID token claims."""
    tenant_id = claims.get("tid")
    email = (
        claims.get("preferred_username")
        or claims.get("upn")
        or claims.get("email")
        or claims.get("unique_name")
    )
    display_name = (
        claims.get("name") or email or claims.get("oid") or "Authenticated User"
    )
    user_id = email or claims.get("oid") or claims.get("sub")

    if not user_id:
        raise ConfigError(
            "Microsoft Entra ID token is missing a stable user identifier."
        )

    roles_claim = claims.get("roles") or ()
    if isinstance(roles_claim, str):
        roles = (roles_claim,)
    else:
        roles = tuple(str(role) for role in roles_claim)

    return AuthContext(
        user_id=str(user_id),
        display_name=str(display_name),
        tenant_id=str(tenant_id) if tenant_id else None,
        email=str(email) if email else None,
        roles=roles,
        authenticated=True,
        auth_source="entra_id",
    )


def _get_or_create_auth_flow() -> dict[str, Any]:
    """Create or reuse the MSAL auth code flow for the current session."""
    existing_flow = st.session_state.get(AUTH_FLOW_SESSION_KEY)
    if isinstance(existing_flow, dict) and existing_flow.get("auth_uri"):
        _cache_auth_flow(existing_flow)
        return existing_flow

    missing_azure_vars = SecureCredentialManager.get_missing_azure_credentials(
        include_client_secret=True
    )
    if missing_azure_vars:
        raise ConfigError(
            "Missing required Azure auth environment variables: "
            + ", ".join(missing_azure_vars)
        )

    app = build_msal_app(allow_common_fallback=not AUTH_STRICT_TENANT)

    # Streamlit callback handling currently reads query parameters from the
    # redirected URL. Using form_post would require a separate HTTP POST
    # callback handler, which this app does not yet expose.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"response_mode='form_post' is recommended.*",
            category=UserWarning,
            module=r"msal\.oauth2cli\.oauth2",
        )
        flow = app.initiate_auth_code_flow(
            scopes=get_login_scopes(),
            redirect_uri=AUTH_REDIRECT_URI,
            response_mode="query",
        )

    st.session_state[AUTH_FLOW_SESSION_KEY] = flow
    _cache_auth_flow(flow)
    return flow


def _try_complete_authentication() -> AuthContext | None:
    """Attempt to complete an auth callback if code and state are present."""
    params = _get_query_params()
    if "code" not in params and "error" not in params:
        return None

    if params.get("error"):
        description = params.get("error_description") or params["error"]
        st.session_state[AUTH_ERROR_SESSION_KEY] = description
        logger.warning("Authentication callback returned an error: %s", description)
        _clear_auth_query_params()
        return None

    callback_state = params.get("state")

    flow = st.session_state.get(AUTH_FLOW_SESSION_KEY)
    if not isinstance(flow, dict):
        flow = _get_cached_auth_flow(callback_state)
        if isinstance(flow, dict):
            st.session_state[AUTH_FLOW_SESSION_KEY] = flow

    if not isinstance(flow, dict):
        st.session_state[AUTH_ERROR_SESSION_KEY] = (
            "Authentication session expired before callback completed. "
            "Please try again."
        )
        _clear_auth_query_params()
        return None

    try:
        app = build_msal_app(allow_common_fallback=not AUTH_STRICT_TENANT)
        result = app.acquire_token_by_auth_code_flow(flow, auth_response=params)
    except Exception as error:  # pylint: disable=broad-except
        logger.error("Failed to complete Microsoft login: %s", sanitise_error(error))
        st.session_state[AUTH_ERROR_SESSION_KEY] = (
            "Authentication could not be completed. Please try again."
        )
        _clear_auth_query_params()
        return None

    _clear_auth_query_params()
    st.session_state.pop(AUTH_FLOW_SESSION_KEY, None)
    _remove_cached_auth_flow(callback_state)

    if not isinstance(result, dict):
        st.session_state[AUTH_ERROR_SESSION_KEY] = (
            "Authentication returned an invalid response."
        )
        return None

    if result.get("error"):
        description = result.get("error_description") or result.get("error")
        st.session_state[AUTH_ERROR_SESSION_KEY] = str(description)
        logger.warning("Authentication failed: %s", description)
        return None

    id_token_claims = result.get("id_token_claims") or {}
    auth_context = _build_auth_context_from_claims(id_token_claims)
    st.session_state.pop(AUTH_ERROR_SESSION_KEY, None)
    return _store_auth_context(auth_context)


def _render_microsoft_sign_in_button(
    auth_uri: str,
    label: str,
    *,
    padding: str,
    font_size: str = "0.95rem",
) -> None:
    """Render a Microsoft-branded sign-in button that keeps the current tab."""
    escaped_auth_uri = html.escape(auth_uri, quote=True)
    escaped_label = html.escape(label)
    max_width = f"{MICROSOFT_SIGN_IN_BUTTON_MAX_WIDTH_PX}px"

    st.markdown(
        f'<a href="{escaped_auth_uri}" target="_self" '
        'style="display: flex; justify-content: center; text-decoration: none;">'
        '<span role="button" style="width: min(100%, '
        f"{max_width}); max-width: {max_width}; min-height: 41px; "
        "display: inline-flex; "
        "align-items: center; justify-content: center; gap: 0.75rem; "
        f"padding: {padding}; box-sizing: border-box; background-color: #ffffff; "
        "color: #1f1f1f; border: 1px solid #8a8886; border-radius: 0.125rem; "
        "cursor: pointer; text-decoration: none; "
        f'font-size: {font_size}; font-weight: 600; line-height: 1.25;">'
        '<span style="display: inline-flex; align-items: center; '
        'justify-content: center; width: 16px; height: 16px; flex-shrink: 0;">'
        f"{MICROSOFT_LOGO_SVG}"
        "</span>"
        f"<span>{escaped_label}</span>"
        "</span></a>",
        unsafe_allow_html=True,
    )


def render_auth_sidebar() -> None:
    """Render authentication status and controls in the sidebar."""
    auth_context = get_auth_context()

    with st.sidebar:
        st.markdown("---")
        st.subheader("Access")

        if auth_context.authenticated:
            st.success(f"Signed in as {auth_context.display_name}")
            if auth_context.email:
                st.caption(auth_context.email)
            if auth_context.roles:
                st.caption(f"Roles: {', '.join(auth_context.roles)}")

            if st.button("Sign out", key="logout_button", use_container_width=True):
                logout()
            return

        if ALLOW_ANONYMOUS_DEV and not REQUIRE_AUTH:
            st.info("Running without sign-in")
        else:
            st.warning("Authentication required")

        try:
            flow = _get_or_create_auth_flow()
        except ConfigError as error:
            st.caption(str(error))
            return

        _render_microsoft_sign_in_button(
            flow["auth_uri"],
            "Sign in with Microsoft",
            padding="0.5rem",
        )


def logout() -> None:
    """Sign out the current user from the application session."""
    _clear_auth_context()
    _clear_auth_query_params()
    st.rerun()


def ensure_authentication() -> AuthContext:
    """Resolve the current user and block access when authentication is mandatory."""
    existing_context = get_auth_context()
    if existing_context.authenticated:
        return existing_context

    callback_context = _try_complete_authentication()
    if callback_context is not None:
        st.rerun()

    auth_error = st.session_state.get(AUTH_ERROR_SESSION_KEY)

    if not authentication_required() and ALLOW_ANONYMOUS_DEV:
        if existing_context.is_anonymous:
            return _store_auth_context(ANONYMOUS_AUTH_CONTEXT)
        return existing_context

    st.markdown("<div style='padding-top: 12vh;'></div>", unsafe_allow_html=True)
    _, centre_column, _ = st.columns([1, 2, 1])

    with centre_column:
        st.markdown(
            "<h2 style='text-align: center; margin-bottom: 0.5rem;'>"
            "Sign in required"
            "</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; margin-bottom: 1.5rem;'>"
            "This application now requires Microsoft Entra ID authentication "
            "before data access is allowed."
            "</p>",
            unsafe_allow_html=True,
        )

        if auth_error:
            st.error(str(auth_error))

        try:
            flow = _get_or_create_auth_flow()
        except ConfigError as error:
            st.error(str(error))
            st.stop()

        _render_microsoft_sign_in_button(
            flow["auth_uri"],
            "Sign in with Microsoft",
            padding="0.75rem",
            font_size="1rem",
        )
    st.stop()
