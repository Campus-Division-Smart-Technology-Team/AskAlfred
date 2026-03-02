# -*- coding: utf-8 -*-
import logging
import re
from typing import Optional
from textblob import TextBlob

from query_preprocessors.base_preprocessor import BasePreprocessor
from building import (
    BuildingCacheManager,
    BUILDING_ALIASES_CACHE,
    BUILDING_NAMES_CACHE,
    normalise_building_name,
)
from log_sanitiser import sanitise_error


class SpellCheckPreprocessor(BasePreprocessor):
    """
    Optional spell checker using TextBlob.

    Safety/UX rules:
      - Enabled by default.
      - Skips when TextBlob isn't available (no hard dependency).
      - Protects building names, business terms, and common domain tokens
        from being "corrected".
    """

    # Module-level cache of the import to satisfy type checkers
    _TextBlob: Optional[type] = None

    def __init__(self) -> None:
        super().__init__()
        self.enabled = True  # always on; protect building/domain tokens
        self.logger = logging.getLogger(self.__class__.__name__)
        self.corrections_made = 0

        # Tokens we never want TextBlob to "correct"
        self.protected_tokens = {
            "fra", "fras", "bms", "ahu", "hvac", "iq4", "o&m",
            "planon", "ppm", "ppm’s", "ppm's"
            "goodbye",
        }
        self.protected_short_tokens = {
            "how", "what", "when", "where", "which", "who", "why",
            "do", "does", "did", "is", "are", "was", "were",
            "can", "could", "would", "should", "will", "shall",
            "bye",
        }
        self.min_protected_token_len = 4

    def _build_protected_tokens(self, context) -> list[str]:
        tokens = set(self.protected_tokens)
        min_len = self.min_protected_token_len

        def _add_token(value: str) -> None:
            if not isinstance(value, str):
                return
            v = value.strip()
            if len(v) < min_len:
                return
            tokens.add(v)

        # Ensure cache is ready and protect known aliases + canonical names
        BuildingCacheManager.ensure_initialised()
        if BUILDING_ALIASES_CACHE:
            for alias in BUILDING_ALIASES_CACHE.keys():
                _add_token(alias)
                _add_token(normalise_building_name(alias))
        if BUILDING_NAMES_CACHE:
            for canonical in BUILDING_NAMES_CACHE.values():
                _add_token(canonical)
                _add_token(normalise_building_name(canonical))

        # Protect detected building names (single or multiple)
        if getattr(context, "building", None):
            _add_token(context.building)
            _add_token(normalise_building_name(context.building))
        for b in getattr(context, "buildings", []) or []:
            _add_token(b)
            _add_token(normalise_building_name(b))

        # Protect detected business terms
        for term in getattr(context, "business_terms", []) or []:
            if isinstance(term, dict):
                t = term.get("term")
                if t:
                    _add_token(t)

        # Always protect common short question/auxiliary tokens.
        tokens.update(self.protected_short_tokens)

        # Filter out empties and sort longest-first to avoid partial overlaps
        return sorted(
            {t for t in tokens if isinstance(t, str) and t.strip()},
            key=len,
            reverse=True,
        )

    def _protect_text(self, text: str, tokens: list[str]) -> tuple[str, list[tuple[str, str]]]:
        """Replace protected tokens with placeholders; return new text + mapping."""
        replacements: list[tuple[str, str]] = []
        protected = text
        counter = 0

        for token in tokens:
            pattern = re.compile(re.escape(token), re.IGNORECASE)

            def _repl(match):
                nonlocal counter
                placeholder = f"ZXQPROT{counter}ZXQ"
                counter += 1
                replacements.append((placeholder, match.group(0)))
                return placeholder

            protected = pattern.sub(_repl, protected)

        return protected, replacements

    def _restore_text(self, text: str, replacements: list[tuple[str, str]]) -> str:
        restored = text
        for placeholder, original in replacements:
            restored = restored.replace(placeholder, original)
        return restored

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
        """
        if not self.enabled:
            return False

        if self._lazy_import_textblob() is None:
            # If it's not installed, just stay quiet and disable ourselves
            self.logger.debug("TextBlob not available; skipping spell check.")
            self.enabled = False
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
            # Compute a candidate correction while protecting tokens
            protected_tokens = self._build_protected_tokens(context)
            protected_text, replacements = self._protect_text(
                context.query, protected_tokens)
            corrected = str(textblob_cls(protected_text).correct())
            corrected = self._restore_text(corrected, replacements)

            if corrected != context.query:
                self.corrections_made += 1

                # Keep originals in cache for traceability
                context.add_to_cache("original_query", context.query)
                context.add_to_cache("spell_corrected", True)

                old_query = context.query
                context.update_query(corrected)

                self.logger.info(
                    "Corrected query: '%s' -> '%s'", old_query, corrected
                )
            else:
                context.add_to_cache("spell_corrected", False)

        except Exception as e:
            # Never fail the pipeline because of spell check
            self.logger.error(
                "SpellCheckPreprocessor failed: %s", sanitise_error(e), exc_info=False)
            context.add_to_cache("spell_corrected", False)
