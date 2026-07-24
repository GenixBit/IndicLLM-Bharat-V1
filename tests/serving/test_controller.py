from __future__ import annotations

from bharat.serving.auth import AuthConfig
from bharat.serving.controller import ServingController
from bharat.serving.metrics import ServingMetrics
from bharat.serving.rate_limit import RateLimitConfig
from bharat.serving.streaming import StreamRequest


class TestServingController:
    def _make_controller(
        self,
        metrics: ServingMetrics | None = None,
    ) -> ServingController:
        return ServingController(
            auth_config=AuthConfig(valid_api_keys=("test-key-001",)),
            rate_limit_config=RateLimitConfig(max_requests=5, window_seconds=60.0),
            metrics=metrics,
        )

    def test_valid_request_returns_events(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hello")
        events = controller.handle_request(request=request, api_key="test-key-001")
        assert len(events) >= 2
        assert events[0].event_type == "text_delta"
        assert events[-1].event_type == "done"

    def test_events_remain_ordered(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hi")
        events = controller.handle_request(request=request, api_key="test-key-001")
        for i, event in enumerate(events[:-1]):
            assert event.index == i

    def test_auth_failure_returns_error_events(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hello")
        events = controller.handle_request(request=request, api_key="invalid-key")
        assert len(events) == 2
        assert events[0].event_type == "error"
        assert events[0].error == "Invalid API key"
        assert events[1].event_type == "done"
        assert events[1].finish_reason == "error"

    def test_missing_auth_returns_error_events(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hello")
        events = controller.handle_request(request=request, api_key=None)
        assert events[0].event_type == "error"
        assert events[0].error == "Missing API key"

    def test_rate_limit_rejection_returns_error_events(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hello")
        for _ in range(5):
            controller.handle_request(request=request, api_key="test-key-001")

        events = controller.handle_request(request=request, api_key="test-key-001")
        assert events[0].event_type == "error"
        assert events[0].error == "Rate limit exceeded"
        assert events[1].event_type == "done"
        assert events[1].finish_reason == "error"

    def test_controller_records_metrics_on_success(self) -> None:
        metrics = ServingMetrics()
        controller = self._make_controller(metrics=metrics)
        request = StreamRequest(prompt="Hello")
        controller.handle_request(request=request, api_key="test-key-001")

        snap = metrics.snapshot()
        assert snap.requests_started >= 1
        assert snap.requests_completed >= 1
        assert snap.streaming_events_emitted >= 2
        assert snap.total_duration_ms > 0.0

    def test_controller_records_auth_failure_metrics(self) -> None:
        metrics = ServingMetrics()
        controller = self._make_controller(metrics=metrics)
        request = StreamRequest(prompt="Hello")
        controller.handle_request(request=request, api_key="bad-key")

        snap = metrics.snapshot()
        assert snap.requests_started >= 1
        assert snap.requests_completed >= 1
        assert snap.auth_failures >= 1
        assert snap.streaming_events_emitted == 0

    def test_controller_records_rate_limit_metrics(self) -> None:
        metrics = ServingMetrics()
        controller = self._make_controller(metrics=metrics)
        request = StreamRequest(prompt="Hello")
        for _ in range(5):
            controller.handle_request(request=request, api_key="test-key-001")
        controller.handle_request(request=request, api_key="test-key-001")

        snap = metrics.snapshot()
        assert snap.rate_limit_rejections >= 1

    def test_without_auth_and_rate_limit(self) -> None:
        metrics = ServingMetrics()
        controller = ServingController(metrics=metrics)
        request = StreamRequest(prompt="Hello")
        events = controller.handle_request(request=request)
        assert len(events) >= 2
        assert events[0].event_type == "text_delta"

    def test_deterministic_output(self) -> None:
        controller = self._make_controller()
        request = StreamRequest(prompt="Hi")
        events1 = controller.handle_request(request=request, api_key="test-key-001")
        controller2 = self._make_controller()
        events2 = controller2.handle_request(request=request, api_key="test-key-001")
        assert events1 == events2
