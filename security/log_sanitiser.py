#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log sanitisation utilities to prevent credential/sensitive data leakage.

Provides functions to redact API keys, passwords, tokens, and other
sensitive information from log messages and stack traces.
"""

import logging
import re
from typing import Any

# Patterns for sensitive data that should be redacted
SENSITIVE_PATTERNS = {
    "api_key": r'(?i)(?:api[_-]?key|apikey)[=:\s]*[\'"]?[\w\-]{20,}[\'"]?',
    "pinecone_key": r'(?i)(?:pinecone[_-]?(?:api[_-])?key)[=:\s]*[\'"]?[\w\-]{20,}[\'"]?',
    "openai_key": r'(?i)(?:openai|sk)[_-]?(?:api[_-])?key[=:\s]*[\'"]?sk[\w\-]{40,}[\'"]?',
    "password": r'(?i)(?:password|passwd|pwd)[=:\s]*[\'"]?[^\s\'"]{6,}[\'"]?',
    "token": r'(?i)(?:token|auth[_-]?token|bearer)[=:\s]*[\'"]?[\w\-]{20,}[\'"]?',
    "secret": r'(?i)(?:secret|client[_-]?secret)[=:\s]*[\'"]?[\w\-]{20,}[\'"]?',
    "credentials": r"(?i)(?:credentials|creds)[=:\s]*\{[^}]+\}",
    "connection_string": r'(?i)(?:connection[_-]?string|mongodb|redis)[=:\s]*[\'"]?[^\s\'"]+[\'"]?',
    "azure_tenant": r'(?i)(?:azure[_-]?tenant[_-]?id|tenant[_-]?id)[=:\s]*[\'"]?[\w\-]+[\'"]?',
    "azure_client_id": r'(?i)(?:azure[_-]?client[_-]?id|client[_-]?id)[=:\s]*[\'"]?[\w\-]+[\'"]?',
    "private_key": r"-----BEGIN[^-]*PRIVATE KEY-----[\s\S]*?-----END[^-]*PRIVATE KEY-----",
}

# Replacement texts
REDACTION_MAP = {
    "api_key": "api_key=***REDACTED***",
    "pinecone_key": "pinecone_api_key=***REDACTED***",
    "openai_key": "openai_api_key=***REDACTED***",
    "password": "password=***REDACTED***",
    "token": "token=***REDACTED***",
    "secret": "secret=***REDACTED***",
    "credentials": "credentials=***REDACTED***",
    "connection_string": "connection_string=***REDACTED***",
    "azure_tenant": "azure_tenant_id=***REDACTED***",
    "azure_client_id": "azure_client_id=***REDACTED***",
    "private_key": "[PRIVATE_KEY_REDACTED]",
}


def sanitise_message(message: str) -> str:
    """
    Sanitise a log message by redacting sensitive data.

    Removes API keys, passwords, tokens, and other credentials.

    Args:
        message: The log message to sanitise

    Returns:
        Sanitised message with sensitive data redacted
    """
    if not isinstance(message, str):
        return str(message)

    sanitised = message

    # Apply all redaction patterns
    for key, pattern in SENSITIVE_PATTERNS.items():
        try:
            replacement = REDACTION_MAP.get(key, "***REDACTED***")
            sanitised = re.sub(
                pattern, replacement, sanitised, flags=re.IGNORECASE | re.MULTILINE
            )
        except re.error:
            # Skip broken patterns
            continue

    return sanitised


def sanitise_error(error: Exception) -> str:
    """
    Sanitise an exception message and its details.

    Args:
        error: The exception to sanitise

    Returns:
        Sanitised error message
    """
    error_str = str(error)
    error_type = type(error).__name__

    # Sanitise the message
    sanitised = sanitise_message(error_str)

    # If sanitisation changed the message, include type info
    if sanitised != error_str:
        return f"{error_type}: {sanitised}"

    return f"{error_type}: {sanitised}"


def sanitise_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively sanitise a dictionary by redacting sensitive values.

    Args:
        data: Dictionary to sanitise

    Returns:
        New dictionary with sensitive values redacted
    """
    if not isinstance(data, dict):
        return data

    sanitised = {}
    sensitive_keys = {
        "api_key",
        "apikey",
        "api-key",
        "password",
        "passwd",
        "pwd",
        "token",
        "auth_token",
        "access_token",
        "secret",
        "client_secret",
        "credentials",
        "creds",
        "key",
        "bearer",
        "pinecone_api_key",
        "openai_api_key",
        "connection_string",
        "database_url",
        "tenant_id",
        "client_id",
    }

    for key, value in data.items():
        # Check if key is sensitive
        if key.lower() in sensitive_keys or any(
            sensitive in key.lower() for sensitive in sensitive_keys
        ):
            sanitised[key] = "***REDACTED***"
        elif isinstance(value, dict):
            # Recursively sanitise nested dicts
            sanitised[key] = sanitise_dict(value)
        elif isinstance(value, (list, tuple)):
            # Sanitise lists/tuples of dicts
            sanitised[key] = [
                sanitise_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str):
            # Sanitise string values
            sanitised[key] = sanitise_message(value)
        else:
            sanitised[key] = value

    return sanitised


class SanitisedFormatter:
    """
    Custom logging formatter that automatically sanitises sensitive data.

    Usage:
        formatter = SanitisedFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
    """

    def __init__(self, fmt: str, **kwargs):
        """Initialise formatter with message format."""
        self.formatter = logging.Formatter(fmt, **kwargs)

    def format(self, record: Any) -> str:
        """Format and sanitise log record."""

        # Sanitise the message
        record.msg = sanitise_message(str(record.msg))

        # Sanitise args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = sanitise_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    (
                        sanitise_dict(arg)
                        if isinstance(arg, dict)
                        else sanitise_message(str(arg)) if isinstance(arg, str) else arg
                    )
                    for arg in record.args
                )

        # Sanitise the fully formatted output as well: the base formatter
        # appends exception tracebacks (exc_info/stack_info) after the
        # message, and secrets raised inside exception messages would
        # otherwise bypass redaction.
        return sanitise_message(self.formatter.format(record))
