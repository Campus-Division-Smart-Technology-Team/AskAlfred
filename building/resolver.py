#!/usr/bin/env python3
"""
Building resolver protocol and shared constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BuildingResolution:
    canonical: str
    confidence: float
    source: str


class BuildingResolver(Protocol):
    def resolve(self, filename: str, text: str) -> BuildingResolution: ...
