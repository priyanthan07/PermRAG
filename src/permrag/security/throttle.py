"""
    Failed-login throttle.

    In-process only: each API worker keeps its own counts, so with N workers
    an attacker gets up to N times the limit. That still turns an unlimited
    online guessing attack into a slow one; a shared store (Redis) would be
    needed for an exact limit across workers.
"""

import time
from collections import deque

from permrag.exceptions import TooManyRequestsError

MAX_FAILURES = 5
WINDOW_SECONDS = 300
# Bound memory: forget the oldest keys past this many tracked identities.
MAX_TRACKED_KEYS = 10_000


class LoginThrottle:
    def __init__(
        self,
        max_failures: int = MAX_FAILURES,
        window_seconds: float = WINDOW_SECONDS,
        max_keys: int = MAX_TRACKED_KEYS,
    ) -> None:
        self._max_failures = max_failures
        self._window = window_seconds
        self._max_keys = max_keys
        self._failures: dict[str, deque[float]] = {}

    def _recent(self, key: str, now: float) -> deque[float]:
        attempts = self._failures.get(key, deque())
        while attempts and now - attempts[0] > self._window:
            attempts.popleft()
        return attempts

    def check(self, key: str) -> None:
        """Raise 429 if ``key`` has too many recent failures."""
        if len(self._recent(key, time.monotonic())) >= self._max_failures:
            raise TooManyRequestsError()

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        attempts = self._recent(key, now)
        attempts.append(now)
        self._failures[key] = attempts
        if len(self._failures) > self._max_keys:
            # dicts keep insertion order: drop the longest-tracked key.
            self._failures.pop(next(iter(self._failures)))

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


login_throttle = LoginThrottle()
