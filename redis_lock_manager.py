import logging
import uuid
import time
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Optional, Protocol
import random
from redis import Redis
from alfred_exceptions import DeadlockError
from config import (
    FRA_LOCK_ACQUIRE_SECONDS,
    REDIS_LOCK_KEY_PREFIX,
    REDIS_LOCK_DEFAULT_TTL_MS,
    REDIS_LOCK_RETRY_INTERVAL_S,
    REDIS_LOCK_JITTER_S,
)


class MetricsRecorder(Protocol):
    def increment(self, key: str, amount: int = 1) -> None: ...
    def observe_timing(self, key: str, value: float) -> None: ...


# ============================================================================
# REDIS-BASED BUILDING LOCK (ALTERNATIVE TO IN-MEMORY)
# ============================================================================
_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
else
  return 0
end
"""

_RENEW_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("PEXPIRE", KEYS[1], ARGV[2])
else
  return 0
end
"""


@dataclass(frozen=True)
class RedisLock:
    """
    A distributed lock handle. Only the holder with the correct token can release/renew.
    """
    key: str
    token: str
    ttl_ms: int
    _client: Redis
    _release_script: Any
    _renew_script: Any
    _logger: logging.Logger

    def renew(self, ttl_ms: Optional[int] = None) -> bool:
        """Extend TTL if we still own the lock."""
        ttl = int(ttl_ms or self.ttl_ms)
        try:
            res = self._renew_script(keys=[self.key], args=[
                                     self.token, str(ttl)])
            ok = bool(res)
            if not ok:
                self._logger.warning(
                    "Redis lock renew failed (not owner?): %s", self.key)
            return ok
        except Exception:
            self._logger.exception("Redis lock renew error: %s", self.key)
            return False

    def release(self) -> bool:
        """Release lock if we still own it."""
        try:
            res = self._release_script(keys=[self.key], args=[self.token])
            ok = bool(res)
            if not ok:
                self._logger.warning(
                    "Redis lock release failed (not owner?): %s", self.key)
            return ok
        except Exception:
            self._logger.exception("Redis lock release error: %s", self.key)
            return False


class RedisLockManager:
    """
    Minimal, safe Redis-based distributed locks:
      - Acquire via SET key token NX PX ttl
      - Release/renew guarded by token via Lua

    Recommended usage:
      with lock_manager.lock(building): ...
      with lock_manager.lock_many(buildings): ...
    """

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = REDIS_LOCK_KEY_PREFIX,
        default_ttl_ms: int = REDIS_LOCK_DEFAULT_TTL_MS,
        acquire_timeout_s: float = FRA_LOCK_ACQUIRE_SECONDS,
        retry_interval_s: float = REDIS_LOCK_RETRY_INTERVAL_S,
        jitter_s: float = REDIS_LOCK_JITTER_S,
        logger: Optional[logging.Logger] = None,
        metrics: Optional["MetricsRecorder"] = None,
    ) -> None:
        if client is None:
            raise RuntimeError(
                "redis client not provided")

        self._client = client
        self._key_prefix = key_prefix.rstrip(":")
        self._default_ttl_ms = int(default_ttl_ms)
        self._acquire_timeout_s = float(acquire_timeout_s)
        self._retry_interval_s = float(retry_interval_s)
        self._jitter_s = float(jitter_s)
        self._logger = logger or logging.getLogger(__name__)
        self._metrics = metrics

        # Pre-register scripts for speed and atomicity
        self._release_script = self._client.register_script(_RELEASE_LUA)
        self._renew_script = self._client.register_script(_RENEW_LUA)

    def _start_auto_renewer(
        self,
        locks: list["RedisLock"],
        *,
        ttl_ms: int,
        interval_s: float,
        lock_lost_event: Optional[Event] = None,
    ) -> Event:
        stop_event = Event()

        def _loop() -> None:
            while not stop_event.wait(interval_s):
                for lk in list(locks):
                    ok = lk.renew(ttl_ms)
                    if not ok:
                        self._logger.warning(
                            "Auto-renew stopped; lost lock ownership: %s",
                            lk.key,
                        )
                        self._logger.critical(
                            "LOCK LOST: %s - operations may be unsafe!", lk.key)
                        if lock_lost_event is not None:
                            lock_lost_event.set()
                        stop_event.set()
                        break

        t = Thread(target=_loop, name="redis-lock-renewer", daemon=True)
        t.start()
        return stop_event

    def _key(self, building: str) -> str:
        # Keep it simple/stable; callers should pass canonical building name
        return f"{self._key_prefix}:{building}"

    def acquire(self, building: str, *, ttl_ms: Optional[int] = None) -> RedisLock:
        """
        Blocking acquire with timeout. Raises DeadlockError if cannot acquire in time.
        """
        key = self._key(building)
        token = uuid.uuid4().hex
        ttl = int(ttl_ms or self._default_ttl_ms)

        attempts = 0
        start = time.monotonic()
        deadline = time.monotonic() + self._acquire_timeout_s
        while True:
            try:
                # SET key token NX PX ttl
                ok = self._client.set(name=key, value=token, nx=True, px=ttl)
            except Exception as e:
                # Treat Redis failures as "can't lock" rather than silently proceeding
                raise DeadlockError(
                    f"Redis lock acquire failed for {building}: {e}") from e

            if ok:
                elapsed = time.monotonic() - start
                if self._metrics is not None:
                    self._metrics.increment("fra_lock_acquire_total")
                    self._metrics.increment(
                        "fra_lock_acquire_attempts_total", attempts + 1)
                    if attempts > 0:
                        self._metrics.increment("fra_lock_contended_total")
                    self._metrics.observe_timing(
                        "fra_lock_acquire_wait_seconds", elapsed)
                return RedisLock(
                    key=key,
                    token=token,
                    ttl_ms=ttl,
                    _client=self._client,
                    _release_script=self._release_script,
                    _renew_script=self._renew_script,
                    _logger=self._logger,
                )

            if time.monotonic() >= deadline:
                raise DeadlockError(
                    f"Could not acquire Redis lock for {building} within "
                    f"{self._acquire_timeout_s:.1f}s (key={key})."
                )

            attempts += 1
            time.sleep(self._retry_interval_s +
                       random.random() * self._jitter_s)

    def release(self, lock: RedisLock) -> bool:
        return lock.release()

    def lock(
        self,
        building: str,
        *,
        ttl_ms: Optional[int] = None,
        auto_renew: bool = False,
        renew_interval_s: Optional[float] = None,
        renew_ttl_ms: Optional[int] = None,
        lock_lost_event: Optional[Event] = None,
    ):
        """
        Context manager for a single building lock.
        If the critical section can exceed the TTL, enable auto_renew.
        """
        manager = self

        class _Ctx:
            def __init__(self):
                self._lock: Optional[RedisLock] = None
                self._stop: Optional[Event] = None

            def __enter__(self) -> RedisLock:
                self._lock = manager.acquire(building, ttl_ms=ttl_ms)
                if auto_renew and self._lock is not None:
                    ttl = int(renew_ttl_ms or self._lock.ttl_ms)
                    interval = float(renew_interval_s or max(
                        1.0, (ttl / 1000.0) / 3.0))
                    self._stop = manager._start_auto_renewer(
                        [self._lock],
                        ttl_ms=ttl,
                        interval_s=interval,
                        lock_lost_event=lock_lost_event,
                    )
                return self._lock

            def __exit__(self, exc_type, exc, tb) -> None:
                if self._stop is not None:
                    self._stop.set()
                if self._lock is not None:
                    manager.release(self._lock)

        return _Ctx()

    def lock_many(
        self,
        buildings: Iterable[str],
        *,
        ttl_ms: Optional[int] = None,
        auto_renew: bool = False,
        renew_interval_s: Optional[float] = None,
        renew_ttl_ms: Optional[int] = None,
        lock_lost_event: Optional[Event] = None,
    ):
        """
        Context manager that acquires multiple building locks in sorted order
        to avoid deadlocks when two workers lock overlapping sets.
        If the critical section can exceed the TTL, enable auto_renew.
        """
        manager = self
        bldgs = sorted({b for b in buildings if b})  # dedupe + stable order

        class _Ctx:
            def __init__(self):
                self._locks: list[RedisLock] = []
                self._stop: Optional[Event] = None

            def __enter__(self) -> list[RedisLock]:
                try:
                    for b in bldgs:
                        self._locks.append(manager.acquire(b, ttl_ms=ttl_ms))
                except Exception:
                    # Release any acquired locks before re-raising.
                    for lk in reversed(self._locks):
                        manager.release(lk)
                    self._locks = []
                    raise

                if auto_renew and self._locks:
                    ttl = int(renew_ttl_ms or self._locks[0].ttl_ms)
                    interval = float(renew_interval_s or max(
                        1.0, (ttl / 1000.0) / 3.0))
                    self._stop = manager._start_auto_renewer(
                        self._locks,
                        ttl_ms=ttl,
                        interval_s=interval,
                        lock_lost_event=lock_lost_event,
                    )
                return self._locks

            def __exit__(self, exc_type, exc, tb) -> None:
                if self._stop is not None:
                    self._stop.set()
                # Release in reverse order
                for lk in reversed(self._locks):
                    manager.release(lk)

        return _Ctx()


class DryRunRedisLockManager:
    """No-op lock manager for dry-run mode."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def lock(self, building: str, *_, **__):
        manager = self

        class _Ctx:
            def __enter__(self) -> None:
                manager._logger.info(
                    "[DRY-RUN] Would acquire lock for %s", building)
                return None

            def __exit__(self, exc_type, exc, tb) -> None:
                manager._logger.info(
                    "[DRY-RUN] Would release lock for %s", building)

        return _Ctx()

    def lock_many(self, buildings: Iterable[str], *_, **__):
        manager = self
        bldgs = sorted({b for b in buildings if b})

        class _Ctx:
            def __enter__(self) -> list[None]:
                manager._logger.info(
                    "[DRY-RUN] Would acquire locks for %d buildings", len(bldgs))
                return [None for _ in bldgs]

            def __exit__(self, exc_type, exc, tb) -> None:
                manager._logger.info(
                    "[DRY-RUN] Would release locks for %d buildings", len(bldgs))

        return _Ctx()
