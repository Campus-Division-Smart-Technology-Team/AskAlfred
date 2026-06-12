#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis-based rate limiting for Alfred - Prevents DoS attacks and resource exhaustion.

Supports:
- Query rate limiting (queries per minute per user)
- API call limiting (external service calls)
- File processing limiting (concurrent file operations)
- Fallback to in-memory limiting if Redis unavailable
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Optional

from core.alfred_exceptions import RateLimitError

logger = logging.getLogger(__name__)


# ===========================================================================
# RATE LIMITING CONSTANTS
# ===========================================================================

# Query rate limits
QUERY_RATE_LIMIT_PER_MINUTE = 30  # Queries per minute per user
QUERY_RATE_LIMIT_WINDOW = 60  # seconds

# API call rate limits (external service calls)
API_CALL_RATE_LIMIT_PER_MINUTE = 100  # API calls per minute per user
API_CALL_RATE_LIMIT_WINDOW = 60  # seconds

# File processing limits
FILE_PROCESSING_CONCURRENT_LIMIT = 5  # Max concurrent file operations
FILE_PROCESSING_RATE_LIMIT = 10  # Max file operations per minute

# Burst limits (allow temporary spikes with stricter recovery)
BURST_ALLOWANCE = 1.2  # Allow 20% burst above normal rate

# ===========================================================================
# RATE LIMITER INTERFACE
# ===========================================================================


class RateLimiterBackend(ABC):
    """Abstract base class for rate limiter implementations."""

    @abstractmethod
    def is_rate_limited(self, key: str, max_calls: int, window_seconds: int) -> bool:
        """
        Check if a key has exceeded its rate limit.

        Args:
            key: Unique identifier (e.g., 'user_123', 'api_call_user_456')
            max_calls: Maximum allowed calls in window
            window_seconds: Time window in seconds

        Returns:
            True if rate limited, False otherwise
        """

    @abstractmethod
    def get_remaining_calls(self, key: str, max_calls: int, window_seconds: int) -> int:
        """
        Get remaining calls allowed for a key in current window.

        Args:
            key: Unique identifier
            max_calls: Maximum allowed calls
            window_seconds: Time window in seconds

        Returns:
            Number of remaining calls (can be negative if over limit)
        """

    @abstractmethod
    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """
        Get Unix timestamp when rate limit resets for a key.

        Args:
            key: Unique identifier
            window_seconds: Time window in seconds

        Returns:
            Unix timestamp when limit resets
        """

    @abstractmethod
    def acquire_lease(self, key: str, duration_seconds: int) -> bool:
        """
        Try to acquire an exclusive lease for a resource (for concurrency control).

        Args:
            key: Resource identifier
            duration_seconds: How long to hold the lease

        Returns:
            True if lease acquired, False if already held by another process
        """

    @abstractmethod
    def release_lease(self, key: str) -> bool:
        """
        Release a lease for a resource.

        Args:
            key: Resource identifier

        Returns:
            True if lease released, False if not held
        """


# ===========================================================================
# IN-MEMORY IMPLEMENTATION (Fallback / Dev)
# ===========================================================================


class InMemoryRateLimiter(RateLimiterBackend):
    """In-memory rate limiter - fast but not suitable for multi-process deployments."""

    def __init__(self):
        self._query_timestamps: dict[str, list[float]] = {}
        self._leases: dict[str, float] = {}  # key -> expiry_time

    def is_rate_limited(self, key: str, max_calls: int, window_seconds: int) -> bool:
        """Check if key is rate limited."""
        current_time = time.time()
        window_start = current_time - window_seconds

        if key not in self._query_timestamps:
            self._query_timestamps[key] = []

        # Clean old timestamps
        self._query_timestamps[key] = [
            ts for ts in self._query_timestamps[key] if ts > window_start
        ]

        if len(self._query_timestamps[key]) >= max_calls:
            logger.warning(
                "Rate limit exceeded for key %s: %d calls in %d seconds",
                key,
                len(self._query_timestamps[key]),
                window_seconds,
            )
            return True

        # Record the call
        self._query_timestamps[key].append(current_time)
        return False

    def get_remaining_calls(self, key: str, max_calls: int, window_seconds: int) -> int:
        """Get remaining calls in window."""
        current_time = time.time()
        window_start = current_time - window_seconds

        if key not in self._query_timestamps:
            return max_calls

        # Count valid timestamps
        valid_calls = sum(1 for ts in self._query_timestamps[key] if ts > window_start)
        return max(0, max_calls - valid_calls)

    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """Get when the rate limit resets."""
        if key not in self._query_timestamps or not self._query_timestamps[key]:
            return time.time()

        oldest_call = min(self._query_timestamps[key])
        return oldest_call + window_seconds

    def acquire_lease(self, key: str, duration_seconds: int) -> bool:
        """Try to acquire a lease."""
        current_time = time.time()

        # Check if lease exists and is still valid
        if key in self._leases and self._leases[key] > current_time:
            return False

        # Acquire the lease
        self._leases[key] = current_time + duration_seconds
        return True

    def release_lease(self, key: str) -> bool:
        """Release a lease."""
        if key not in self._leases:
            return False

        del self._leases[key]
        return True


# ===========================================================================
# REDIS IMPLEMENTATION
# ===========================================================================


class RedisRateLimiter(RateLimiterBackend):
    """Redis-backed rate limiter - suitable for distributed deployments."""

    def __init__(self, redis_client):
        """
        Initialize Redis rate limiter.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.lua_increment = None  # Lazy-load Lua script

    def _get_increment_script(self):
        """Get Lua script for atomic increment with window."""
        if self.lua_increment is None:
            # Lua script for atomic increment with TTL
            # Returns [current_count, is_limited]
            # NOTE: Lua `false` converts to a Redis nil, which truncates the
            # returned array — the flag must be returned as 1/0, never as a
            # boolean, or callers see a single-element reply.
            self.lua_increment = self.redis.register_script("""
                local key = KEYS[1]
                local window = tonumber(ARGV[1])
                local max_calls = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])

                -- Remove old entries outside window
                redis.call('zremrangebyscore', key, '-inf', now - window)

                -- Get current count
                local current = redis.call('zcard', key)

                -- Check if limited
                local is_limited = current >= max_calls

                -- Add current timestamp if not limited
                if not is_limited then
                    redis.call('zadd', key, now, now)
                    redis.call('expire', key, window + 10)  -- Expire after window
                end

                return {current, is_limited and 1 or 0}
            """)
        return self.lua_increment

    def is_rate_limited(self, key: str, max_calls: int, window_seconds: int) -> bool:
        """Check if key is rate limited using Redis."""
        try:
            script = self._get_increment_script()
            current_time = time.time()

            result = script(
                keys=[f"rate_limit:{key}"],
                args=[window_seconds, max_calls, current_time],
            )

            is_limited = result[1]
            if is_limited:
                logger.warning(
                    "Rate limit exceeded for key %s: %d calls in %d seconds",
                    key,
                    result[0],
                    window_seconds,
                )
            return bool(is_limited)

        except Exception as e:
            logger.error("Redis rate limit check failed: %s - falling back to allow", e)
            # Fail open (allow the operation) rather than blocking
            return False

    def get_remaining_calls(self, key: str, max_calls: int, window_seconds: int) -> int:
        """Get remaining calls in window."""
        try:
            current_time = time.time()
            window_start = current_time - window_seconds

            # Count valid entries in window
            self.redis.zremrangebyscore(f"rate_limit:{key}", "-inf", window_start)
            current_count = self.redis.zcard(f"rate_limit:{key}")

            return max(0, max_calls - current_count)

        except Exception as e:
            logger.error("Redis remaining calls check failed: %s", e)
            return max_calls  # Fail open - assume unlimited

    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """Get when rate limit resets."""
        try:
            # Get the earliest timestamp in the sorted set
            scores = self.redis.zrange(f"rate_limit:{key}", 0, 0, withscores=True)
            if scores:
                oldest_time = scores[0][1]
                return oldest_time + window_seconds
            else:
                return time.time()

        except Exception as e:
            logger.error("Redis reset time check failed: %s", e)
            return time.time() + window_seconds

    def acquire_lease(self, key: str, duration_seconds: int) -> bool:
        """Try to acquire a lease using Redis."""
        try:
            # Use SET with EX and NX for atomic compare-and-set
            lease_key = f"lease:{key}"
            result = self.redis.set(
                lease_key,
                str(time.time()),
                ex=duration_seconds,
                nx=True,  # Only set if not exists
            )
            return bool(result)

        except Exception as e:
            logger.error("Redis lease acquisition failed: %s", e)
            return True  # Fail open

    def release_lease(self, key: str) -> bool:
        """Release a lease."""
        try:
            lease_key = f"lease:{key}"
            return bool(self.redis.delete(lease_key))

        except Exception as e:
            logger.error("Redis lease release failed: %s", e)
            return True  # Fail open


# ===========================================================================
# GLOBAL RATE LIMITER MANAGER
# ===========================================================================


class RateLimiterManager:
    """Manages global rate limiting with automatic backend selection."""

    def __init__(self):
        self._backend: RateLimiterBackend | None = None
        self._use_redis = False

    def initialise(self, redis_client: Optional[Any] = None):
        """
        Initialise rate limiter with appropriate backend.

        Args:
            redis_client: Optional Redis client. If None, uses in-memory backend.
        """
        if self._backend is not None:
            return

        if redis_client:
            try:
                # Test Redis connection
                redis_client.ping()
                self._backend = RedisRateLimiter(redis_client)
                self._use_redis = True
                logger.info("Using Redis-backed rate limiting")
            except Exception as e:
                logger.warning(
                    "Redis unavailable for rate limiting: %s - using in-memory", e
                )
                self._backend = InMemoryRateLimiter()
        else:
            self._backend = InMemoryRateLimiter()
            logger.info("Using in-memory rate limiting (dev mode)")

    def _ensure_initialised(self):
        """Ensure backend is initialised."""
        if self._backend is None:
            self._backend = InMemoryRateLimiter()

    def is_rate_limited(self, key: str, max_calls: int, window_seconds: int) -> bool:
        """Check if key is rate limited."""
        self._ensure_initialised()
        assert self._backend is not None
        return self._backend.is_rate_limited(key, max_calls, window_seconds)

    def get_remaining_calls(self, key: str, max_calls: int, window_seconds: int) -> int:
        """Get remaining calls allowed."""
        self._ensure_initialised()
        assert self._backend is not None
        return self._backend.get_remaining_calls(key, max_calls, window_seconds)

    def get_reset_time(self, key: str, window_seconds: int) -> float:
        """Get rate limit reset time."""
        self._ensure_initialised()
        assert self._backend is not None
        return self._backend.get_reset_time(key, window_seconds)

    def acquire_lease(self, key: str, duration_seconds: int) -> bool:
        """Try to acquire a resource lease."""
        self._ensure_initialised()
        assert self._backend is not None
        return self._backend.acquire_lease(key, duration_seconds)

    def release_lease(self, key: str) -> bool:
        """Release a resource lease."""
        self._ensure_initialised()
        assert self._backend is not None
        return self._backend.release_lease(key)


# Global instance
_rate_limiter_manager = RateLimiterManager()


def initialise_rate_limiter(redis_client: Optional[object] = None):
    """
    Initialise the global rate limiter.

    Args:
        redis_client: Optional Redis client instance
    """
    _rate_limiter_manager.initialise(redis_client)


# ===========================================================================
# RATE LIMITING DECORATORS
# ===========================================================================


def rate_limit(max_calls: Optional[int] = None, window_seconds: int = 60):
    """
    Decorator to apply rate limiting to a function.

    Args:
        max_calls: Maximum calls allowed per window (if None, uses appropriate default)
        window_seconds: Time window in seconds

    Example:
        @rate_limit(max_calls=30, window_seconds=60)
        def process_query(user_id, query):
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract user_id from arguments
            user_id = kwargs.get("user_id") or (
                args[0] if args and isinstance(args[0], str) else "anonymous"
            )

            actual_max_calls = max_calls or QUERY_RATE_LIMIT_PER_MINUTE

            limiter_key = f"{func.__name__}:{user_id}"
            if _rate_limiter_manager.is_rate_limited(
                limiter_key, actual_max_calls, window_seconds
            ):
                reset_time = _rate_limiter_manager.get_reset_time(
                    limiter_key, window_seconds
                )
                retry_after = max(0, int(reset_time - time.time()))
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds."
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def acquire_resource_lease(resource_id: str, duration_seconds: int = 300):
    """
    Acquire an exclusive lease on a resource.

    Used to prevent concurrent operations on the same resource.

    Args:
        resource_id: Unique resource identifier
        duration_seconds: How long to hold the lease

    Returns:
        True if lease acquired, False otherwise
    """
    return _rate_limiter_manager.acquire_lease(resource_id, duration_seconds)


def release_resource_lease(resource_id: str):
    """
    Release a lease on a resource.

    Args:
        resource_id: Unique resource identifier

    Returns:
        True if lease released
    """
    return _rate_limiter_manager.release_lease(resource_id)


# ===========================================================================
# CONVENIENCE FUNCTIONS
# ===========================================================================


def check_query_rate_limit(user_id: str) -> bool:
    """
    Check if user can make a new query.

    Args:
        user_id: Unique user identifier

    Returns:
        True if NOT rate limited (query allowed), False if rate limited
    """
    return not _rate_limiter_manager.is_rate_limited(
        f"query:{user_id}", QUERY_RATE_LIMIT_PER_MINUTE, QUERY_RATE_LIMIT_WINDOW
    )


def check_api_call_rate_limit(user_id: str) -> bool:
    """
    Check if user can make external API calls.

    Args:
        user_id: Unique user identifier

    Returns:
        True if NOT rate limited, False if rate limited
    """
    return not _rate_limiter_manager.is_rate_limited(
        f"api_call:{user_id}",
        API_CALL_RATE_LIMIT_PER_MINUTE,
        API_CALL_RATE_LIMIT_WINDOW,
    )


def check_file_processing_limit() -> bool:
    """
    Check if file processing queue is at capacity.

    Returns:
        True if file processing available, False if at limit
    """
    return not _rate_limiter_manager.is_rate_limited(
        "file_processing",
        FILE_PROCESSING_CONCURRENT_LIMIT,
        1,  # Check per second for concurrent limit
    )


def get_query_remaining_calls(user_id: str) -> int:
    """Get remaining queries allowed for user."""
    return _rate_limiter_manager.get_remaining_calls(
        f"query:{user_id}", QUERY_RATE_LIMIT_PER_MINUTE, QUERY_RATE_LIMIT_WINDOW
    )


def get_query_reset_time(user_id: str) -> float:
    """Get Unix timestamp when query limit resets for user."""
    return _rate_limiter_manager.get_reset_time(
        f"query:{user_id}", QUERY_RATE_LIMIT_WINDOW
    )
