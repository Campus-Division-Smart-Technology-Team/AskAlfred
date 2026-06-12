#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microsoft Authentication Library (MSAL) configuration for Azure AD.
Handles secure credential loading from credential manager.
"""

import logging
import os
from typing import Optional

import msal

from auth.credential_manager import SecureCredentialManager
from core.alfred_exceptions import ConfigError
from core.env_bootstrap import load_local_env
from security.log_sanitiser import sanitise_error

load_local_env()

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = ["email", "User.Read"]
MSAL_RESERVED_SCOPES = {"openid", "offline_access", "profile"}


def _get_azure_config() -> dict[str, str]:
    """
    Load Azure AD configuration from credential manager.

    Does NOT store credentials in memory - retrieves fresh from environment each time.
    """
    try:
        return SecureCredentialManager.get_azure_config()
    except KeyError as e:
        raise ConfigError(
            "Azure credentials not configured. "
            "Please set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET "
            "environment variables."
        ) from e


def _get_authority(allow_common_fallback: bool = False) -> str:
    """
    Build MSAL authority URL.

    Uses credential manager to get tenant ID securely.
    """
    try:
        tenant_id = SecureCredentialManager.get_azure_tenant_id()
        return f"https://login.microsoftonline.com/{tenant_id}"
    except KeyError as exc:
        if allow_common_fallback:
            return "https://login.microsoftonline.com/common"
        raise ConfigError(
            "Azure tenant ID not configured. Please set AZURE_TENANT_ID for "
            "single-tenant authentication."
        ) from exc


def get_login_scopes() -> list[str]:
    """Return the configured login scopes for Entra ID sign-in."""
    configured_scopes = os.getenv("AUTH_SCOPES", "")
    raw_scopes = (
        [scope.strip() for scope in configured_scopes.split(",") if scope.strip()]
        if configured_scopes.strip()
        else list(DEFAULT_SCOPES)
    )

    filtered_scopes = [
        scope for scope in raw_scopes if scope.lower() not in MSAL_RESERVED_SCOPES
    ]

    if configured_scopes.strip():
        removed_scopes = [
            scope for scope in raw_scopes if scope.lower() in MSAL_RESERVED_SCOPES
        ]
        if removed_scopes:
            logger.info(
                "AUTH_SCOPES contained reserved MSAL scopes and they were ignored: %s",
                ", ".join(removed_scopes),
            )

    return filtered_scopes or list(DEFAULT_SCOPES)


SCOPES = get_login_scopes()


def build_msal_app(
    cache: Optional[object] = None,
    allow_common_fallback: bool = False,
) -> msal.ConfidentialClientApplication:
    """
    Build MSAL ConfidentialClientApplication for Azure AD authentication.

    Credentials are loaded on-demand from environment, not cached in memory.

    Args:
        cache: Optional token cache for credential storage
        allow_common_fallback: Whether to fall back to the common authority
            when no tenant ID is configured

    Returns:
        Configured MSAL ConfidentialClientApplication instance

    Raises:
        ConfigError: If Azure credentials are not available
    """
    try:
        config = _get_azure_config()

        return msal.ConfidentialClientApplication(
            client_id=config.get("AZURE_CLIENT_ID"),
            authority=_get_authority(allow_common_fallback=allow_common_fallback),
            client_credential=config.get("AZURE_CLIENT_SECRET"),
            token_cache=cache,
        )
    except ConfigError:
        raise
    except Exception as e:
        logger.error("Failed to build MSAL application: %s", sanitise_error(e))
        raise ConfigError("Failed to initialize Azure AD authentication") from e
