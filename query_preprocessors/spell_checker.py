#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Optional
from textblob import TextBlob

from query_preprocessors.base_preprocessor import BasePreprocessor


class SpellCheckPreprocessor(BasePreprocessor):
    """
    Optional spell checker using TextBlob.

    Safety/UX rules:
      - Disabled by default (enable explicitly if you want it).
      - Skips when TextBlob isn't available (no hard dependency).
      - Skips when we've already detected a building or business terms,
        to avoid "correcting" proper nouns or acronyms.
      - Protects common domain tokens from being "corrected".
    """

    # Module-level cache of the import to satisfy type checkers
    _TextBlob: Optional[type] = None

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False  # opt-in only
        self.logger = logging.getLogger(self.__class__.__name__)
        self.corrections_made = 0

        # Tokens we never want TextBlob to "correct"
        self.protected_tokens = {
            "fra", "fras", "bms", "ahu", "hvac", "iq4", "o&m",
            "planon", "ppm", "ppm’s", "ppm's"
        }

    # -------- internal helpers -------- #

    @classmethod
    def _lazy_import_textblob(cls) -> Optional[type]:
        """Import TextBlob once; return None if not installed."""
        if cls._TextBlob is not None:
            return cls._TextBlob
        try:
            cls._TextBlob = TextBlob
        except Exception:
            cls._TextBlob = None
        return cls._TextBlob

    def should_run(self, context) -> bool:
        """
        Run only when:
          - enabled
          - TextBlob is available
          - we haven't already detected specific entities that could be harmed
        """
        if not self.enabled:
            return False

        if self._lazy_import_textblob() is None:
            # If it's not installed, just stay quiet and disable ourselves
            self.logger.debug("TextBlob not available; skipping spell check.")
            self.enabled = False
            return False

        # If a building was already detected, don't risk "correcting" it
        if context.get_from_cache("building_detected"):
            return False

        # If business terms (acronyms) were detected, skip
        if getattr(context, "business_terms", None):
            return False

        # If the query contains any protected tokens, skip
        q = (context.query or "").lower()
        if any(tok in q for tok in self.protected_tokens):
            return False

        return True

    # -------- main hook -------- #

    def process(self, context) -> None:
        if not self.should_run(context):
            return

        textblob_cls = self._lazy_import_textblob()
        if textblob_cls is None:
            # Extra guard for type checkers / runtime
            return

        try:
            # Compute a candidate correction
            corrected = str(textblob_cls(context.query).correct())

            if corrected != context.query:
                self.corrections_made += 1

                # Keep originals in cache for traceability
                context.add_to_cache("original_query", context.query)
                context.add_to_cache("spell_corrected", True)

                # Update the query in-place (no update_query() method exists)
                old_query = context.query
                context.query = corrected

                self.logger.info(
                    "Corrected query: '%s' -> '%s'", old_query, corrected
                )
            else:
                context.add_to_cache("spell_corrected", False)

        except Exception as e:
            # Never fail the pipeline because of spell check
            self.logger.error(
                "SpellCheckPreprocessor failed: %s", e, exc_info=True)
            context.add_to_cache("spell_corrected", False)
