from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ServingMetrics:
    requests_started: int = 0
    requests_completed: int = 0
    auth_failures: int = 0
    rate_limit_rejections: int = 0
    streaming_events_emitted: int = 0
    streaming_errors: int = 0
    total_duration_ms: float = 0.0

    def increment_requests_started(self, count: int = 1) -> None:
        self.requests_started += count

    def increment_requests_completed(self, count: int = 1) -> None:
        self.requests_completed += count

    def increment_auth_failures(self, count: int = 1) -> None:
        self.auth_failures += count

    def increment_rate_limit_rejections(self, count: int = 1) -> None:
        self.rate_limit_rejections += count

    def increment_streaming_events_emitted(self, count: int = 1) -> None:
        self.streaming_events_emitted += count

    def increment_streaming_errors(self, count: int = 1) -> None:
        self.streaming_errors += count

    def add_duration_ms(self, duration_ms: float) -> None:
        self.total_duration_ms += duration_ms

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            requests_started=self.requests_started,
            requests_completed=self.requests_completed,
            auth_failures=self.auth_failures,
            rate_limit_rejections=self.rate_limit_rejections,
            streaming_events_emitted=self.streaming_events_emitted,
            streaming_errors=self.streaming_errors,
            total_duration_ms=self.total_duration_ms,
        )


@dataclass(frozen=True)
class MetricsSnapshot:
    requests_started: int = 0
    requests_completed: int = 0
    auth_failures: int = 0
    rate_limit_rejections: int = 0
    streaming_events_emitted: int = 0
    streaming_errors: int = 0
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "auth_failures": self.auth_failures,
            "rate_limit_rejections": self.rate_limit_rejections,
            "streaming_events_emitted": self.streaming_events_emitted,
            "streaming_errors": self.streaming_errors,
            "total_duration_ms": self.total_duration_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
