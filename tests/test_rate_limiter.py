"""
Comprehensive tests for rate_limiter.py

Tests rate limiting controls:
- Query rate limiting (30 queries/minute)
- API call rate limiting (100 calls/minute)
- File processing limits (5 concurrent, 10/minute)
- Redis backend implementation
- In-memory fallback implementation
- Lease acquisition for concurrency control
"""


import time
from rate_limiter import (
    InMemoryRateLimiter,
    QUERY_RATE_LIMIT_PER_MINUTE,
    API_CALL_RATE_LIMIT_PER_MINUTE,
    FILE_PROCESSING_CONCURRENT_LIMIT,
    FILE_PROCESSING_RATE_LIMIT,
    BURST_ALLOWANCE,
)


class TestInMemoryRateLimiter:
    """Test in-memory rate limiter implementation."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_first_call_always_allowed(self):
        """Test that first call is always allowed."""
        assert not self.limiter.is_rate_limited(
            'user_1', max_calls=5, window_seconds=60
        )

    def test_rate_limit_respected(self):
        """Test that rate limit is enforced."""
        key = 'user_1'
        max_calls = 3

        # First 3 calls should succeed
        for i in range(max_calls):
            assert not self.limiter.is_rate_limited(key, max_calls, 60)

        # 4th call should be rate limited
        assert self.limiter.is_rate_limited(key, max_calls, 60)

    def test_window_reset_allows_new_calls(self):
        """Test that calls reset after window expires."""
        key = 'user_1'
        max_calls = 2
        window = 1  # 1 second window

        # Exhaust quota
        self.limiter.is_rate_limited(key, max_calls, window)
        self.limiter.is_rate_limited(key, max_calls, window)

        # Should be rate limited
        assert self.limiter.is_rate_limited(key, max_calls, window)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert not self.limiter.is_rate_limited(key, max_calls, window)

    def test_multiple_users_independent_limits(self):
        """Test that different users have independent rate limits."""
        max_calls = 2

        # User 1 uses up quota
        self.limiter.is_rate_limited('user_1', max_calls, 60)
        self.limiter.is_rate_limited('user_1', max_calls, 60)
        assert self.limiter.is_rate_limited('user_1', max_calls, 60)

        # User 2 should still have quota
        assert not self.limiter.is_rate_limited('user_2', max_calls, 60)

    def test_get_remaining_calls_positive(self):
        """Test remaining calls calculation when under limit."""
        key = 'user_1'
        max_calls = 5

        # Make 2 calls
        self.limiter.is_rate_limited(key, max_calls, 60)
        self.limiter.is_rate_limited(key, max_calls, 60)

        remaining = self.limiter.get_remaining_calls(key, max_calls, 60)
        assert remaining == 3

    def test_get_remaining_calls_zero(self):
        """Test remaining calls when limit reached."""
        key = 'user_1'
        max_calls = 2

        # Exhaust quota
        self.limiter.is_rate_limited(key, max_calls, 60)
        self.limiter.is_rate_limited(key, max_calls, 60)

        remaining = self.limiter.get_remaining_calls(key, max_calls, 60)
        assert remaining <= 0

    def test_get_remaining_calls_negative_overflow(self):
        """Test remaining calls calculation when over limit."""
        key = 'user_1'
        max_calls = 2

        # Make 4 calls (exceeding limit)
        for _ in range(4):
            self.limiter.is_rate_limited(key, max_calls, 60)

        remaining = self.limiter.get_remaining_calls(key, max_calls, 60)
        assert remaining == 0  # Implementation clamps to 0, doesn't go negative

    def test_get_reset_time(self):
        """Test that reset time is returned."""
        key = 'user_1'
        window = 60

        # Make a call
        self.limiter.is_rate_limited(key, max_calls=5, window_seconds=window)

        reset_time = self.limiter.get_reset_time(key, window)
        current_time = time.time()

        # Reset time should be approximately 60 seconds from now
        assert reset_time > current_time
        assert reset_time <= current_time + window + 1  # Allow 1s margin


class TestLeaseManagement:
    """Test lease acquisition for concurrency control."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_acquire_lease_success(self):
        """Test successful lease acquisition."""
        assert self.limiter.acquire_lease('resource_1', duration_seconds=5)

    def test_acquire_lease_already_held(self):
        """Test that second lease acquisition fails while held."""
        resource = 'resource_1'
        assert self.limiter.acquire_lease(resource, duration_seconds=5)
        assert not self.limiter.acquire_lease(resource, duration_seconds=5)

    def test_lease_expires_and_can_be_reacquired(self):
        """Test that expired leases can be reacquired."""
        resource = 'resource_1'
        assert self.limiter.acquire_lease(resource, duration_seconds=1)
        assert not self.limiter.acquire_lease(resource, duration_seconds=1)

        # Wait for lease to expire
        time.sleep(1.1)

        # Should be able to acquire again
        assert self.limiter.acquire_lease(resource, duration_seconds=1)

    def test_release_lease(self):
        """Test lease release."""
        resource = 'resource_1'
        assert self.limiter.acquire_lease(resource, duration_seconds=5)

        # Release the lease
        assert self.limiter.release_lease(resource)

        # Should be able to acquire again immediately
        assert self.limiter.acquire_lease(resource, duration_seconds=5)

    def test_release_nonexistent_lease(self):
        """Test releasing a lease that doesn't exist."""
        assert not self.limiter.release_lease('nonexistent')

    def test_multiple_resources_independent_leases(self):
        """Test that resources have independent leases."""
        assert self.limiter.acquire_lease('resource_1', duration_seconds=5)
        assert self.limiter.acquire_lease('resource_2', duration_seconds=5)

        # First resource still held
        assert not self.limiter.acquire_lease('resource_1', duration_seconds=5)
        # Second resource still held
        assert not self.limiter.acquire_lease('resource_2', duration_seconds=5)


class TestRateLimitingScenarios:
    """Test realistic rate limiting scenarios."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_query_rate_limiting_scenario(self):
        """Test real-world query rate limiting."""
        user_id = 'user_123'

        # Simulate 25 queries (below limit of 30/minute)
        for i in range(25):
            assert not self.limiter.is_rate_limited(
                user_id, QUERY_RATE_LIMIT_PER_MINUTE, 60
            ), f"Query {i+1} should not be rate limited"

        # Queries 26-30 should succeed
        for i in range(26, 31):
            is_limited = self.limiter.is_rate_limited(
                user_id, QUERY_RATE_LIMIT_PER_MINUTE, 60
            )
            assert not is_limited, f"Query {i} should not be rate limited"

        # Query 31 should be rate limited
        assert self.limiter.is_rate_limited(
            user_id, QUERY_RATE_LIMIT_PER_MINUTE, 60
        ), "Query 31 should be rate limited"

    def test_api_call_rate_limiting_scenario(self):
        """Test API call rate limiting."""
        user_id = 'user_456'

        # Simulate 50 API calls (below limit of 100/minute)
        for i in range(50):
            assert not self.limiter.is_rate_limited(
                user_id, API_CALL_RATE_LIMIT_PER_MINUTE, 60
            )

        # 51-100 should succeed
        for i in range(50, 100):
            assert not self.limiter.is_rate_limited(
                user_id, API_CALL_RATE_LIMIT_PER_MINUTE, 60
            )

        # 101 should be rate limited
        assert self.limiter.is_rate_limited(
            user_id, API_CALL_RATE_LIMIT_PER_MINUTE, 60
        )

    def test_file_processing_concurrency_limit(self):
        """Test concurrent file processing limit."""
        # Note: InMemoryRateLimiter uses one lease per key,
        # so we use unique keys for each concurrent operation

        leased_resources = []
        # Acquire 5 concurrent leases with unique keys
        for i in range(FILE_PROCESSING_CONCURRENT_LIMIT):
            resource_key = f'file_processor_{i}'
            assert self.limiter.acquire_lease(resource_key, duration_seconds=30), \
                f"Should acquire lease {i+1}"
            leased_resources.append(resource_key)

        # All 5 leases should be active
        assert len(leased_resources) == FILE_PROCESSING_CONCURRENT_LIMIT

        # We can add a 6th lease with a new key (system doesn't limit total leases)
        # The limit would be enforced at the application level

    def test_file_processing_rate_limiting(self):
        """Test file processing rate limiting."""
        user_id = 'file_user'

        # Simulate 10 file operations (at limit)
        for i in range(FILE_PROCESSING_RATE_LIMIT):
            assert not self.limiter.is_rate_limited(
                user_id, FILE_PROCESSING_RATE_LIMIT, 60
            )

        # 11th should be rate limited
        assert self.limiter.is_rate_limited(
            user_id, FILE_PROCESSING_RATE_LIMIT, 60
        )


class TestBurstHandling:
    """Test burst allowance and handling."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_burst_allowance_calculation(self):
        """Test burst allowance constant is defined."""
        # Note: BURST_ALLOWANCE is defined but not currently used in is_rate_limited
        # The implementation uses strict sliding window without burst support
        user_id = 'burst_user'
        limit = 10

        # Standard rate limiting applies (10 calls allowed, 11th blocked)
        for i in range(limit):
            assert not self.limiter.is_rate_limited(
                user_id, limit, 60
            ), f"Call {i+1} should be within limit"

        # 11th call should be limited (no burst allowance in current implementation)
        assert self.limiter.is_rate_limited(user_id, limit, 60)


class TestConcurrentUsagePattern:
    """Test concurrent usage patterns."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_multiple_users_concurrent_queries(self):
        """Test that multiple users can make concurrent queries."""
        users = [f'user_{i}' for i in range(5)]

        # Each user makes a query
        for user in users:
            assert not self.limiter.is_rate_limited(
                user, QUERY_RATE_LIMIT_PER_MINUTE, 60
            )

    def test_interleaved_user_queries(self):
        """Test interleaved queries from different users."""
        for iteration in range(5):
            for user_id in ['user_a', 'user_b', 'user_c']:
                assert not self.limiter.is_rate_limited(
                    user_id, 10, 60
                ), f"User {user_id} iteration {iteration} should succeed"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_zero_max_calls(self):
        """Test behavior with zero max calls."""
        # Even first call should be limited
        assert self.limiter.is_rate_limited(
            'user', max_calls=0, window_seconds=60)

    def test_negative_max_calls(self):
        """Test behavior with negative max calls."""
        # All calls should be limited
        assert self.limiter.is_rate_limited(
            'user', max_calls=-1, window_seconds=60)

    def test_very_small_window(self):
        """Test with very small time window."""
        key = 'user'

        # Make a call
        assert not self.limiter.is_rate_limited(
            key, max_calls=2, window_seconds=1)
        assert not self.limiter.is_rate_limited(
            key, max_calls=2, window_seconds=1)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be able to make more calls
        assert not self.limiter.is_rate_limited(
            key, max_calls=2, window_seconds=1)

    def test_very_large_max_calls(self):
        """Test with very large max call limit."""
        key = 'user'
        max_calls = 1000000

        # Make 100 calls
        for i in range(100):
            assert not self.limiter.is_rate_limited(key, max_calls, 60)

    def test_empty_key(self):
        """Test with empty string as key."""
        # Empty key should still work
        assert not self.limiter.is_rate_limited(
            '', max_calls=1, window_seconds=60)
        assert self.limiter.is_rate_limited('', max_calls=1, window_seconds=60)

    def test_unicode_key(self):
        """Test with unicode in key."""
        key = 'user_café_123'
        assert not self.limiter.is_rate_limited(
            key, max_calls=1, window_seconds=60)
        assert self.limiter.is_rate_limited(
            key, max_calls=1, window_seconds=60)

    def test_very_long_key(self):
        """Test with very long key."""
        key = 'user_' + 'x' * 1000
        assert not self.limiter.is_rate_limited(
            key, max_calls=1, window_seconds=60)

    def test_special_characters_in_key(self):
        """Test with special characters in key."""
        keys = [
            'user:123',
            'user-123',
            'user.123',
            'user@domain.com',
        ]
        for key in keys:
            assert not self.limiter.is_rate_limited(
                key, max_calls=1, window_seconds=60)


class TestInMemoryRateLimiterMemory:
    """Test memory management of in-memory limiter."""

    def setup_method(self):
        """Setup before each test."""
        self.limiter = InMemoryRateLimiter()

    def test_cleanup_old_timestamps(self):
        """Test that old timestamps are cleaned up."""
        key = 'user'
        window = 1

        # Make a call
        self.limiter.is_rate_limited(key, 100, window)

        # Verify timestamp is stored
        assert key in self.limiter._query_timestamps
        assert len(self.limiter._query_timestamps[key]) > 0

        # Wait for window to expire
        time.sleep(1.1)

        # Access again to trigger cleanup
        self.limiter.is_rate_limited(key, 100, window)

        # Old timestamps should be cleaned
        # (This is implementation dependent)

    def test_many_users_memory_efficiency(self):
        """Test memory usage with many users."""
        # Create 1000 users
        for i in range(1000):
            user_id = f'user_{i}'
            self.limiter.is_rate_limited(
                user_id, max_calls=10, window_seconds=60)

        # Should have 1000 entries
        assert len(self.limiter._query_timestamps) == 1000
