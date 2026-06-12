#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authentication context models for Alfred."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuthContext:
    """Represents the current user identity for a request."""

    user_id: str
    display_name: str
    tenant_id: str | None = None
    email: str | None = None
    roles: tuple[str, ...] = field(default_factory=tuple)
    authenticated: bool = False
    auth_source: str = "anonymous"

    @property
    def is_anonymous(self) -> bool:
        """Return True when the current user is not authenticated."""
        return not self.authenticated


ANONYMOUS_AUTH_CONTEXT = AuthContext(
    user_id="anonymous",
    display_name="Anonymous",
    authenticated=False,
    auth_source="anonymous",
)
