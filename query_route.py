#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QueryRoute - Represents the routing decision for a query.

Contains the selected handler and optional metadata used during routing.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryRoute:
    handler: Any                           # Instance of a handler
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
