#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handler for conversational queries (greetings, about, etc.).
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random
from query_types import QueryType
from query_context import QueryContext
from query_result import QueryResult
from emojis import EMOJI_BUILDING, EMOJI_GORILLA, EMOJI_FIRE, EMOJI_MAINTENANCE

from .base_handler import BaseQueryHandler


class ConversationalHandler(BaseQueryHandler):
    """
    Handles natural conversational queries:
    greetings, gratitude, farewells, small-talk, identity questions.
    """

    def __init__(self):
        super().__init__()
        self.query_type = QueryType.CONVERSATIONAL
        self.priority = 1

        # Patterns grouped by conversation category
        self.pattern_groups = {
            "greeting": [
                re.compile(r"^hey[\s,]+alfred", re.IGNORECASE),
                re.compile(r"^(hi|hello|hey)[!., ]*$", re.IGNORECASE),
                re.compile(
                    r"^(good\s+(morning|afternoon|evening))$", re.IGNORECASE),
                re.compile(r"^(hi|hello)[\s,]+alfred", re.IGNORECASE),
            ],
            "gratitude": [
                re.compile(r"^thanks?[!., ]*$", re.IGNORECASE),
                re.compile(r"^thank\s+you[!., ]*$", re.IGNORECASE),
                re.compile(r"\bappreciate\s+it\b", re.IGNORECASE),
                re.compile(r"^cheers[!., ]*$", re.IGNORECASE),
            ],
            "farewell": [
                re.compile(r"^(bye|goodbye)[!., ]*$", re.IGNORECASE),
                re.compile(r"^(see\s+you|take\s+care)[!., ]*$", re.IGNORECASE),
            ],
            "about": [
                re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
                re.compile(r"\bwhat\s+are\s+you\b", re.IGNORECASE),
                re.compile(r"\bwhat\s+can\s+you\s+do\b", re.IGNORECASE),
                re.compile(r"\btell\s+me\s+about\s+yourself\b", re.IGNORECASE),
            ],
            "smalltalk": [
                re.compile(r"^how\s+are\s+you\??$", re.IGNORECASE),
                re.compile(r"^how[' ]s\s+it\s+going\??$", re.IGNORECASE),
            ],
        }

        # Response sets
        self.responses = {
            "greeting": [
                f"Hello! I'm Alfred {EMOJI_GORILLA}, your helpful assistant at the University of Bristol. I can help you find information about:\n\n {EMOJI_BUILDING} Building Management Systems (BMS)\n {EMOJI_FIRE} Fire Risk Assessments (FRAs)\n {EMOJI_MAINTENANCE} Maintenance Requests and Jobs\n\nWhat would you like to know today?",
                f"Hi there! I'm Alfred {EMOJI_GORILLA}, ready to help you search through our knowledge bases. Feel free to ask me about BMS {EMOJI_BUILDING}, FRAs {EMOJI_FIRE} and maintenance requests and jobs {EMOJI_MAINTENANCE}. How can I assist you?",
                f"Hello! Alfred here {EMOJI_GORILLA}, your University of Bristol assistant. I have access to information about Building Management Systems and Fire Risk Assessments and maintenance requests and jobs. What can I help you with?"
            ],
            "gratitude": [
                "You're welcome.",
                "Happy to help.",
                "Any time.",
            ],
            "farewell": [
                "Goodbye!",
                "See you next time.",
                "Take care.",
            ],
            "about": [
                "I'm Alfred, a university assistant built to help you explore building data, maintenance info, and documents.",
                "I'm Alfred — here to answer questions about campus buildings, maintenance records, and property information.",
                "I'm Alfred, your campus information guide.",
            ],
            "smalltalk": [
                "I'm operating smoothly — ready when you are.",
                "Doing well. What can I help you with?",
                "All systems running. What’s on your mind?",
            ],
        }

    def can_handle(self, context: QueryContext) -> bool:
        """Check each category’s patterns."""
        query = context.query.strip().lower()

        for ctype, patterns in self.pattern_groups.items():
            for p in patterns:
                if p.search(query):
                    context.add_to_cache("conversation_type", ctype)
                    return True
        return False

    def handle(self, context: QueryContext) -> QueryResult:
        """Return a conversational response."""
        ctype = context.get_from_cache("conversation_type", "greeting")
        response = random.choice(self.responses.get(ctype, ["Hello!"]))

        return QueryResult(
            query=context.query,
            answer=response,
            results=[],
            query_type=self.query_type.value,
            handler_used="ConversationalHandler",
            metadata={"conversation_type": ctype},
        )
