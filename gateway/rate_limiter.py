import asyncio
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, status

from gateway.auth import require_api_key
from gateway.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class FixedWindowRateLimiter:
    """Per-key fixed-window in-memory rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # maps key -> (request_count, window_start_monotonic)
        self._windows: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """Return True if the key is within its current rate-limit window."""
        async with self._lock:
            count, window_start = self._windows[key]
            now = time.monotonic()
            if now - window_start >= self.window_seconds:
                self._windows[key] = (1, now)
                return True
            if count >= self.max_requests:
                return False
            self._windows[key] = (count + 1, window_start)
            return True


_limiter = FixedWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


async def check_rate_limit(api_key: str = Depends(require_api_key)) -> str:
    """Raise 429 if the authenticated key has exceeded its rate limit."""
    if not await _limiter.is_allowed(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit of {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS}s exceeded"
            ),
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )
    return api_key
