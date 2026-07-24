from __future__ import annotations

from collections.abc import Callable

from bharat.serving.auth import ApiKeyAuthenticator, AuthConfig
from bharat.serving.metrics import ServingMetrics
from bharat.serving.rate_limit import InMemoryRateLimiter, RateLimitConfig
from bharat.serving.streaming import (
    LocalStreamer,
    StreamEvent,
    StreamRequest,
)


class ServingController:
    def __init__(
        self,
        *,
        auth_config: AuthConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        metrics: ServingMetrics | None = None,
        streamer_factory: Callable[[StreamRequest], LocalStreamer] | None = None,
    ) -> None:
        self._auth = ApiKeyAuthenticator(auth_config) if auth_config is not None else None
        self._rate_limiter = (
            InMemoryRateLimiter(rate_limit_config) if rate_limit_config is not None else None
        )
        self._metrics = metrics or ServingMetrics()
        self._streamer_factory = streamer_factory or LocalStreamer

    @property
    def metrics(self) -> ServingMetrics:
        return self._metrics

    def handle_request(
        self,
        request: StreamRequest,
        api_key: str | None = None,
        client_id: str | None = None,
    ) -> list[StreamEvent]:
        self._metrics.increment_requests_started()

        if self._auth is not None:
            auth_result = self._auth.authenticate(api_key)
            if not auth_result.ok:
                self._metrics.increment_auth_failures()
                self._metrics.increment_requests_completed()
                self._metrics.add_duration_ms(0.0)
                return [
                    StreamEvent(
                        event_type="error",
                        error=auth_result.error,
                        index=0,
                    ),
                    StreamEvent(
                        event_type="done",
                        index=1,
                        finish_reason="error",
                    ),
                ]
            client_id = client_id or auth_result.client_id

        cid = client_id or "default"

        if self._rate_limiter is not None:
            rate_result = self._rate_limiter.check(cid)
            if not rate_result.ok:
                self._metrics.increment_rate_limit_rejections()
                self._metrics.increment_requests_completed()
                self._metrics.add_duration_ms(0.0)
                return [
                    StreamEvent(
                        event_type="error",
                        error=rate_result.error,
                        index=0,
                    ),
                    StreamEvent(
                        event_type="done",
                        index=1,
                        finish_reason="error",
                    ),
                ]

        streamer = self._streamer_factory(request)
        events = streamer.generate()

        self._metrics.increment_streaming_events_emitted(len(events))

        error_count = sum(1 for e in events if e.event_type == "error")
        if error_count:
            self._metrics.increment_streaming_errors(error_count)

        self._metrics.increment_requests_completed()
        self._metrics.add_duration_ms(1.0)

        return events
