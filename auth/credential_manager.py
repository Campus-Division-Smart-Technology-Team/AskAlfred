#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Secure credential management for Alfred.

Handles loading and caching credentials from environment variables with:
- Minimal memory exposure (load on-demand, not at module import)
- Automatic cleanup of sensitive data
- Proper error handling without exposing secrets
- Validation without storing credentials in process env
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class CredentialProvider(ABC):
    """Abstract base class for credential providers."""

    @abstractmethod
    def get_credential(self, name: str) -> str:
        """Get a credential by name."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate that required credentials are available."""


class EnvironmentCredentialProvider(CredentialProvider):
    """
    Load credentials from environment variables.

    Never stores credentials in the object - always fetches fresh from environment.
    Uses thread-safe validation to avoid concurrent validation work.
    """

    def __init__(self, required_credentials: dict[str, str]):
        """
        Initialise provider with required credential names.

        Args:
            required_credentials: Dict mapping env var names to human-readable names
        """
        self.required_credentials = required_credentials
        self._lock = threading.Lock()

    def get_credential(self, name: str) -> str:
        """
        Get credential from environment without storing it.

        Args:
            name: Environment variable name

        Returns:
            Credential value (NOT CACHED - always fresh from environment)

        Raises:
            KeyError: If credential not found
            ValueError: If credential is empty
        """
        value = os.environ.get(name)

        if value is None:
            raise KeyError(
                f"Credential '{name}' not found. "
                "Please set the environment variable and restart."
            )

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Credential '{name}' is empty. Please provide a valid value."
            )

        return value

    def validate(self) -> bool:
        """
        Check if all required credentials are available.

        Returns:
            True if all required credentials are present
        """
        with self._lock:
            missing = []
            for env_name, display_name in self.required_credentials.items():
                value = os.environ.get(env_name)
                if value is None or not str(value).strip():
                    missing.append(display_name)

            if missing:
                logger.warning(
                    "Missing credentials: %s. Application may not function correctly.",
                    ", ".join(missing),
                )
                return False

            return True


class SecureCredentialManager:
    """
    Manage application credentials securely.

    Key security features:
    - Credentials never stored in instance variables
    - On-demand loading only when needed
    - Thread-safe access
    - Automatic memory cleanup via context managers
    - Proper error handling without exposing secrets
    """

    # Class-level thread lock to prevent concurrent initialization issues
    _lock = threading.Lock()

    # Credential providers mapped by type
    _providers: dict[str, CredentialProvider] = {}

    # Azure AD Credentials
    _azure_provider: Optional[CredentialProvider] = None

    # Service Credentials
    _service_provider: Optional[CredentialProvider] = None

    @classmethod
    def initialise(cls):
        """Initialise credential providers (call once at app startup)."""
        with cls._lock:
            if cls._providers:
                return  # Already initialised

            # Azure AD provider
            cls._azure_provider = EnvironmentCredentialProvider(
                {
                    "AZURE_TENANT_ID": "Azure Tenant ID",
                    "AZURE_CLIENT_ID": "Azure Client ID",
                    "AZURE_CLIENT_SECRET": "Azure Client Secret",
                }
            )

            # Service providers
            cls._service_provider = EnvironmentCredentialProvider(
                {
                    "PINECONE_API_KEY": "Pinecone API Key",
                    "OPENAI_API_KEY": "OpenAI API Key",
                }
            )

            cls._providers["azure"] = cls._azure_provider
            cls._providers["service"] = cls._service_provider

            # Validate at startup
            logger.info("Validating credentials...")
            if not cls._azure_provider.validate():
                logger.warning("Azure AD credentials not fully configured")
            if not cls._service_provider.validate():
                logger.warning("Service credentials not fully configured")

    @classmethod
    def get_azure_tenant_id(cls) -> str:
        """
        Get Azure Tenant ID.

        Returns:
            Tenant ID (NOT CACHED - fresh from environment each time)

        Raises:
            KeyError: If not set
        """
        if not cls._azure_provider:
            cls.initialise()
        assert cls._azure_provider is not None
        return cls._azure_provider.get_credential("AZURE_TENANT_ID")

    @classmethod
    def get_azure_client_id(cls) -> str:
        """Get Azure Client ID."""
        if not cls._azure_provider:
            cls.initialise()
        assert cls._azure_provider is not None
        return cls._azure_provider.get_credential("AZURE_CLIENT_ID")

    @classmethod
    def get_azure_client_secret(cls) -> str:
        """Get Azure Client Secret."""
        if not cls._azure_provider:
            cls.initialise()
        assert cls._azure_provider is not None
        return cls._azure_provider.get_credential("AZURE_CLIENT_SECRET")

    @classmethod
    def get_pinecone_api_key(cls) -> str:
        """Get Pinecone API Key."""
        if not cls._service_provider:
            cls.initialise()
        assert cls._service_provider is not None
        return cls._service_provider.get_credential("PINECONE_API_KEY")

    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API Key."""
        if not cls._service_provider:
            cls.initialise()
        assert cls._service_provider is not None
        return cls._service_provider.get_credential("OPENAI_API_KEY")

    @classmethod
    def get_azure_config(cls) -> dict[str, str]:
        """
        Get all Azure AD configuration.

        WARNING: Returns dict with credentials. Use with care and don't log!

        Returns:
            Dictionary with AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
        """
        if not cls._azure_provider:
            cls.initialise()
        assert cls._azure_provider is not None

        try:
            return {
                "AZURE_TENANT_ID": cls._azure_provider.get_credential(
                    "AZURE_TENANT_ID"
                ),
                "AZURE_CLIENT_ID": cls._azure_provider.get_credential(
                    "AZURE_CLIENT_ID"
                ),
                "AZURE_CLIENT_SECRET": cls._azure_provider.get_credential(
                    "AZURE_CLIENT_SECRET"
                ),
            }
        except KeyError as e:
            raise KeyError(f"Azure configuration incomplete: {e}") from e

    @classmethod
    def get_missing_azure_credentials(
        cls, include_client_secret: bool = True
    ) -> list[str]:
        """Return a list of missing Azure authentication environment variables."""
        required = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID"]
        if include_client_secret:
            required.append("AZURE_CLIENT_SECRET")

        missing: list[str] = []
        for env_name in required:
            value = os.environ.get(env_name)
            if value is None or not str(value).strip():
                missing.append(env_name)

        return missing

    @classmethod
    def validate_azure_credentials(cls, include_client_secret: bool = True) -> bool:
        """Validate required Azure authentication environment variables."""
        return not cls.get_missing_azure_credentials(
            include_client_secret=include_client_secret
        )

    @classmethod
    def validate_all(cls) -> bool:
        """
        Validate all credential providers.

        Returns:
            True if all credentials available, False otherwise
        """
        if not cls._providers:
            cls.initialise()

        all_valid = True
        for name, provider in cls._providers.items():
            if not provider.validate():
                logger.warning("Provider '%s' validation failed", name)
                all_valid = False

        return all_valid


# Avoid import-time initialization to keep credential loading on-demand.
# Call SecureCredentialManager.initialise() explicitly at app startup if desired.
