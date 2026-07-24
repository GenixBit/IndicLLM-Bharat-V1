from __future__ import annotations

import time as _time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitConfig:
    max_requests: int = 60
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {self.max_requests}")
        if self.window_seconds <= 0.0:
            raise ValueError(f"window_seconds must be > 0.0, got {self.window_seconds}")


@dataclass(frozen=True)
class RateLimitResult:
    ok: bool
    remaining: int = 0
    reset_after: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.ok and not self.error:
            raise ValueError("Rate-limit rejection must include an error message")
        if self.ok and self.error:
            raise ValueError("Rate-limit success must not include an error message")


class InMemoryRateLimiter:
    def __init__(
        self,
        config: RateLimitConfig,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._time_fn = time_fn or _time.time
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> RateLimitResult:
        now = self._time_fn()
        window_start = now - self._config.window_seconds

        timestamps = self._buckets[client_id]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self._config.max_requests:
            oldest = timestamps[0]
            reset_after = oldest + self._config.window_seconds - now
            return RateLimitResult(
                ok=False,
                remaining=0,
                reset_after=max(reset_after, 0.0),
                error="Rate limit exceeded",
            )

        timestamps.append(now)
        remaining = self._config.max_requests - len(timestamps)
        return RateLimitResult(
            ok=True,
            remaining=remaining,
            reset_after=self._config.window_seconds,
        )
