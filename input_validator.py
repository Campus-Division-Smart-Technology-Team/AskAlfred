#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced input validation for Alfred - Guards against prompt injection,
DoS attacks, and malicious input patterns.

Implements OWASP input validation best practices.
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Optional
import logging

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Prompt injection attack patterns
INJECTION_PATTERNS = [
    # Ignore/override patterns
    r'(?i)(ignore\s+previous|disregard|forget|override|clear\s+context)',
    # System prompt reference patterns
    r'(?i)(system\s+prompt|instructions|you\s+are|you\s+are\s+a|role\s*[=:]|persona\s*[=:])',
    # Jailbreak patterns
    r'(?i)(as\s+an\s+ai|as\s+an\s+assistant|pretend|act\s+like|behave\s+as)',
    # Output manipulation
    r'(?i)(output\s+only|response\s+must|format\s+as|reply\s+with)',
    # Constraint breaking
    r'(?i)(ignore\s+constraints|no\s+constraints|bypass|circumvent)',
    # Hidden instructions
    r'(?i)(hidden\s+message|secret\s+instruction|true\s+purpose)',
]

# Special characters that are suspicious when overrepresented
SUSPICIOUS_SPECIAL_CHARS = set('{}[]|<>\\`$;:')

# Characters allowed in normal queries
ALLOWED_SPECIAL_CHARS = set(',.!?\'"()-&@#%+=/~*')

# Query complexity limits
QUERY_COMPLEXITY_LIMIT = 500  # tokens (rough estimation)
SPECIAL_CHAR_RATIO_LIMIT = 0.3  # Max 30% special characters
SUSPICIOUS_CHAR_RATIO_LIMIT = 0.1  # Max 10% suspicious characters
REPEATED_CHAR_LIMIT = 5  # Max 5 repeated characters in a row

# Rate limiting
RATE_LIMIT_PER_MINUTE = 30  # Default: 30 queries per minute per user
RATE_LIMIT_WINDOW_SECONDS = 60

# Pinecone filter operators - allowed for safe filtering
ALLOWED_FILTER_OPERATORS = {'$eq', '$ne',
                            '$gt', '$gte', '$lt', '$lte', '$in', '$nin'}
DANGEROUS_FILTER_KEYWORDS = {'$where', '$regex', '$text', 'javascript', 'eval'}

# ===========================================================================
# LOGGING
# ===========================================================================
logger = logging.getLogger(__name__)


# ===========================================================================
# INJECTION DETECTION
# ===========================================================================

@lru_cache(maxsize=1000)
def is_injection_attempt(query: str) -> bool:
    """
    Detect potential prompt injection attack patterns.

    Args:
        query: User query string to validate

    Returns:
        True if injection pattern detected, False otherwise
    """
    if not query:
        return False

    # Check against known injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query):
            logger.warning(
                "Potential injection pattern detected: query length=%d", len(query))
            return True

    return False


# ===========================================================================
# SPECIAL CHARACTER VALIDATION
# ===========================================================================

def count_special_characters(query: str) -> tuple[int, int, int]:
    """
    Count different types of special characters in query.

    Returns:
        Tuple of (total_special, suspicious_count, repeated_max)
    """
    special_count = 0
    suspicious_count = 0
    max_repeated = 1
    current_repeated = 1
    last_char = ''

    for char in query:
        if not char.isalnum() and not char.isspace():
            special_count += 1
            if char in SUSPICIOUS_SPECIAL_CHARS:
                suspicious_count += 1
            if char == last_char:
                current_repeated += 1
                max_repeated = max(max_repeated, current_repeated)
            else:
                current_repeated = 1
            last_char = char
        else:
            last_char = ''

    return special_count, suspicious_count, max_repeated


def has_excessive_special_chars(query: str) -> tuple[bool, Optional[str]]:
    """
    Check if query has excessive special characters.

    Returns:
        Tuple of (is_invalid, error_message)
    """
    query_len = len(query)
    special_count, suspicious_count, max_repeated = count_special_characters(
        query)

    # Check special character ratio
    if special_count > query_len * SPECIAL_CHAR_RATIO_LIMIT:
        error = (f"Query contains too many special characters "
                 f"({special_count}/{query_len} = {100*special_count/query_len:.1f}%)")
        logger.warning("Query rejected: %s", error)
        return True, error

    # Check suspicious character ratio
    if suspicious_count > query_len * SUSPICIOUS_CHAR_RATIO_LIMIT:
        error = (f"Query contains suspicious character patterns "
                 f"({suspicious_count} dangerous chars)")
        logger.warning("Query rejected: %s", error)
        return True, error

    # Check for repeated characters (potential DoS)
    if max_repeated > REPEATED_CHAR_LIMIT:
        error = f"Query has too many repeated characters (max {max_repeated})"
        logger.warning("Query rejected: %s", error)
        return True, error

    return False, None


# ===========================================================================
# QUERY VALIDATION
# ===========================================================================

def validate_query_security(
    query: str,
    min_length: int = 2,
    max_length: int = 1000,
) -> tuple[bool, Optional[str]]:
    """
    Validate query for security issues.

    Checks for:
    - Empty/null input
    - Length constraints
    - Prompt injection attempts
    - Excessive special characters
    - Suspicious patterns

    Args:
        query: User query string
        min_length: Minimum query length (default 2)
        max_length: Maximum query length (default 1000)

    Returns:
        Tuple of (is_valid, error_message)
    """

    # Check if empty
    if not query or not query.strip():
        return False, "Please enter a question."

    query = query.strip()

    # Check length constraints
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)."

    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)."

    # Check for prompt injection attempts
    if is_injection_attempt(query):
        return False, "Query contains invalid patterns. Please rephrase your question."

    # Check for excessive special characters
    is_invalid, error_msg = has_excessive_special_chars(query)
    if is_invalid:
        return False, error_msg

    return True, None


# ===========================================================================
# PINECONE FILTER VALIDATION
# ===========================================================================

def sanitise_filter_value(value: str) -> str:
    """
    Sanitise filter value to prevent NoSQL injection in Pinecone filters.

    Args:
        value: Filter value to sanitise

    Returns:
        Sanitised value safe for use in Pinecone filters
    """
    if not isinstance(value, str):
        return str(value)

    # Remove any dangerous characters/operators
    sanitised = value
    for keyword in DANGEROUS_FILTER_KEYWORDS:
        sanitised = re.sub(rf'\b{keyword}\b', '',
                           sanitised, flags=re.IGNORECASE)

    # Escape special regex characters
    sanitised = re.escape(sanitised)

    return sanitised


def validate_filter_operator(operator: str) -> bool:
    """
    Validate that filter operator is in allowed list.

    Args:
        operator: Filter operator (e.g., '$eq', '$in')

    Returns:
        True if operator is allowed, False otherwise
    """
    return operator in ALLOWED_FILTER_OPERATORS


def sanitise_pinecone_filter(filter_dict: dict) -> dict:
    """
    Sanitise Pinecone filter dictionary to prevent injection attacks.

    Args:
        filter_dict: Pinecone filter dictionary

    Returns:
        Sanitised filter dictionary
    """
    if not isinstance(filter_dict, dict):
        return {}

    sanitised = {}

    for key, value in filter_dict.items():
        # Check for dangerous key patterns
        if any(kw in key.lower() for kw in ('javascript', 'eval', 'exec')):
            logger.warning("Dangerous filter key detected: %s", key)
            continue

        # Sanitise based on value type
        if isinstance(value, str):
            sanitised[key] = sanitise_filter_value(value)
        elif isinstance(value, dict):
            # Validate operators in nested filters
            for op, val in value.items():
                if not validate_filter_operator(op):
                    logger.warning("Invalid filter operator: %s", op)
                    continue
                sanitised.setdefault(key, {})[
                    op] = sanitise_filter_value(str(val))
        elif isinstance(value, (int, float, bool)):
            sanitised[key] = value
        elif isinstance(value, list):
            sanitised[key] = [sanitise_filter_value(str(v)) for v in value]

    return sanitised


# ===========================================================================
# BUILDING NAME VALIDATION
# ===========================================================================

def validate_building_name(building_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate building name for safety.

    Args:
        building_name: Building name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not building_name or not building_name.strip():
        return False, "Building name cannot be empty"

    name = building_name.strip()

    # Check length
    if len(name) > 500:
        return False, "Building name too long (max 500 chars)"

    # Check for injection patterns (building names should be simple)
    if is_injection_attempt(name):
        return False, "Building name contains invalid patterns"

    # Check for excessive special characters
    is_invalid, error_msg = has_excessive_special_chars(name)
    if is_invalid:
        return False, error_msg

    return True, None


# ===========================================================================
# CONTENT ESCAPING
# ===========================================================================

def escape_markdown_special_chars(text: str) -> str:
    """
    Escape special markdown characters to prevent injection.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for markdown rendering
    """
    # Escape markdown special characters
    escape_chars = r'\`*_{}[]()#+-.!'
    result = text
    for char in escape_chars:
        result = result.replace(char, f'\\{char}')
    return result


def is_safe_for_markdown(text: str) -> bool:
    """
    Check if text is safe to render in markdown without sanitization.

    Args:
        text: Text to check

    Returns:
        True if text is safe, False if it should be escaped
    """
    # Check for markdown link syntax [text](url)
    if re.search(r'\[.+\]\(.+\)', text):
        return False

    # Check for HTML
    if re.search(r'<[^>]+>', text):
        return False

    # Check for script tags
    if re.search(r'<script|javascript:', text, re.IGNORECASE):
        return False

    return True


# ===========================================================================
# RATE LIMITING SUPPORT
# ===========================================================================

class RateLimitChecker:
    """
    Simple in-memory rate limiter for queries.

    For production, use Redis-based rate limiting.
    """

    def __init__(self):
        self._query_counts = {}  # user_id -> [(timestamp, count)]
        self._lock = {}

    def is_rate_limited(self, user_id: str, max_calls: int = RATE_LIMIT_PER_MINUTE) -> bool:
        """
        Check if user has exceeded rate limit.

        Args:
            user_id: Unique user identifier
            max_calls: Max calls allowed per minute

        Returns:
            True if rate limited, False otherwise
        """
        import time

        current_time = time.time()
        window_start = current_time - RATE_LIMIT_WINDOW_SECONDS

        if user_id not in self._query_counts:
            self._query_counts[user_id] = []

        # Remove old entries
        self._query_counts[user_id] = [
            ts for ts in self._query_counts[user_id]
            if ts > window_start
        ]

        # Check if exceeded
        if len(self._query_counts[user_id]) >= max_calls:
            logger.warning(
                "Rate limit exceeded for user %s: %d calls in %d seconds",
                user_id,
                len(self._query_counts[user_id]),
                RATE_LIMIT_WINDOW_SECONDS
            )
            return True

        # Record this query
        self._query_counts[user_id].append(current_time)
        return False


# Global rate limiter instance
_rate_limiter = RateLimitChecker()


def check_user_rate_limit(user_id: str) -> bool:
    """
    Check if user has exceeded rate limit.

    Args:
        user_id: Unique user identifier

    Returns:
        True if NOT rate limited (query allowed), False if rate limited
    """
    return not _rate_limiter.is_rate_limited(user_id, RATE_LIMIT_PER_MINUTE)


# ===========================================================================
# QUERY SUMMARY (for analytics/logging)
# ===========================================================================

def get_validation_summary(query: str) -> dict:
    """
    Get detailed validation information about a query (for logging/analytics).

    Args:
        query: User query to analyze

    Returns:
        Dictionary with validation details
    """
    special_count, suspicious_count, max_repeated = count_special_characters(
        query)

    return {
        'query_length': len(query),
        'word_count': len(query.split()),
        'special_char_count': special_count,
        'suspicious_char_count': suspicious_count,
        'max_repeated_chars': max_repeated,
        'has_injection_patterns': is_injection_attempt(query),
        'has_excessive_special_chars': has_excessive_special_chars(query)[0],
    }
