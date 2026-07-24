from __future__ import annotations

import pytest

from bharat.serving.rate_limit import InMemoryRateLimiter, RateLimitConfig, RateLimitResult


class TestRateLimitConfig:
    def test_valid(self) -> None:
        config = RateLimitConfig(max_requests=10, window_seconds=30.0)
        assert config.max_requests == 10
        assert config.window_seconds == 30.0

    def test_zero_max_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="max_requests must be >= 1"):
            RateLimitConfig(max_requests=0)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be > 0.0"):
            RateLimitConfig(window_seconds=0.0)

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be > 0.0"):
            RateLimitConfig(window_seconds=-1.0)


class TestRateLimitResult:
    def test_success(self) -> None:
        result = RateLimitResult(ok=True, remaining=9, reset_after=60.0)
        assert result.ok
        assert result.error is None

    def test_failure_with_error(self) -> None:
        result = RateLimitResult(ok=False, error="Rate limit exceeded", reset_after=30.0)
        assert not result.ok
        assert result.error == "Rate limit exceeded"

    def test_failure_without_error_raises(self) -> None:
        with pytest.raises(ValueError, match="must include an error message"):
            RateLimitResult(ok=False)

    def test_success_with_error_raises(self) -> None:
        with pytest.raises(ValueError, match="must not include an error message"):
            RateLimitResult(ok=True, error="Should not happen")


class TestInMemoryRateLimiter:
    def test_allows_under_limit(self) -> None:
        config = RateLimitConfig(max_requests=5, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=lambda: 100.0)
        result = limiter.check("client-1")
        assert result.ok
        assert result.remaining == 4

    def test_rejects_over_limit(self) -> None:
        config = RateLimitConfig(max_requests=3, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=lambda: 100.0)
        for _ in range(3):
            limiter.check("client-1")
        result = limiter.check("client-1")
        assert not result.ok
        assert result.error == "Rate limit exceeded"

    def test_resets_after_window(self) -> None:
        times: list[float] = [100.0]

        def clock() -> float:
            return times[0]

        config = RateLimitConfig(max_requests=2, window_seconds=10.0)
        limiter = InMemoryRateLimiter(config, time_fn=clock)

        assert limiter.check("client-1").ok
        assert limiter.check("client-1").ok
        assert not limiter.check("client-1").ok

        times[0] = 111.0
        result = limiter.check("client-1")
        assert result.ok
        assert result.remaining == 1

    def test_per_client_limits(self) -> None:
        config = RateLimitConfig(max_requests=2, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=lambda: 100.0)

        assert limiter.check("client-a").ok
        assert limiter.check("client-a").ok
        assert not limiter.check("client-a").ok

        assert limiter.check("client-b").ok
        assert limiter.check("client-b").ok

    def test_remaining_count_decreases(self) -> None:
        config = RateLimitConfig(max_requests=5, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=lambda: 100.0)

        assert limiter.check("client-1").remaining == 4
        assert limiter.check("client-1").remaining == 3
        assert limiter.check("client-1").remaining == 2

    def test_remaining_zero_on_rejection(self) -> None:
        config = RateLimitConfig(max_requests=1, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=lambda: 100.0)

        limiter.check("client-1")
        result = limiter.check("client-1")
        assert result.remaining == 0
        assert not result.ok

    def test_deterministic_injected_clock(self) -> None:
        calls: list[float] = []

        def clock() -> float:
            t = 200.0
            calls.append(t)
            return t

        config = RateLimitConfig(max_requests=2, window_seconds=60.0)
        limiter = InMemoryRateLimiter(config, time_fn=clock)
        limiter.check("client-1")
        limiter.check("client-1")
        assert len(calls) == 2
