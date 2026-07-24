from __future__ import annotations

import json

from bharat.serving.metrics import MetricsSnapshot, ServingMetrics


class TestServingMetrics:
    def test_initial_state(self) -> None:
        m = ServingMetrics()
        snap = m.snapshot()
        assert snap.requests_started == 0
        assert snap.requests_completed == 0
        assert snap.auth_failures == 0
        assert snap.rate_limit_rejections == 0
        assert snap.streaming_events_emitted == 0
        assert snap.streaming_errors == 0
        assert snap.total_duration_ms == 0.0

    def test_increment_requests_started(self) -> None:
        m = ServingMetrics()
        m.increment_requests_started()
        assert m.snapshot().requests_started == 1
        m.increment_requests_started(3)
        assert m.snapshot().requests_started == 4

    def test_increment_requests_completed(self) -> None:
        m = ServingMetrics()
        m.increment_requests_completed()
        assert m.snapshot().requests_completed == 1

    def test_increment_auth_failures(self) -> None:
        m = ServingMetrics()
        m.increment_auth_failures()
        assert m.snapshot().auth_failures == 1

    def test_increment_rate_limit_rejections(self) -> None:
        m = ServingMetrics()
        m.increment_rate_limit_rejections()
        assert m.snapshot().rate_limit_rejections == 1

    def test_increment_streaming_events_emitted(self) -> None:
        m = ServingMetrics()
        m.increment_streaming_events_emitted(5)
        assert m.snapshot().streaming_events_emitted == 5

    def test_increment_streaming_errors(self) -> None:
        m = ServingMetrics()
        m.increment_streaming_errors()
        assert m.snapshot().streaming_errors == 1

    def test_add_duration_ms(self) -> None:
        m = ServingMetrics()
        m.add_duration_ms(150.5)
        assert m.snapshot().total_duration_ms == 150.5
        m.add_duration_ms(50.0)
        assert m.snapshot().total_duration_ms == 200.5

    def test_snapshot_isolation(self) -> None:
        m = ServingMetrics()
        snap1 = m.snapshot()
        m.increment_requests_started()
        snap2 = m.snapshot()
        assert snap1.requests_started == 0
        assert snap2.requests_started == 1


class TestMetricsSnapshot:
    def test_to_dict(self) -> None:
        snap = MetricsSnapshot(
            requests_started=10,
            requests_completed=8,
            auth_failures=2,
            rate_limit_rejections=1,
            streaming_events_emitted=50,
            streaming_errors=0,
            total_duration_ms=2500.0,
        )
        d = snap.to_dict()
        assert d["requests_started"] == 10
        assert d["total_duration_ms"] == 2500.0

    def test_to_json_is_deterministic(self) -> None:
        snap = MetricsSnapshot(
            requests_started=1,
            requests_completed=1,
            auth_failures=0,
            rate_limit_rejections=0,
            streaming_events_emitted=5,
            streaming_errors=0,
            total_duration_ms=100.0,
        )
        j1 = snap.to_json()
        j2 = snap.to_json()
        assert j1 == j2

    def test_to_json_is_valid(self) -> None:
        snap = MetricsSnapshot(
            requests_started=3,
            requests_completed=2,
            auth_failures=1,
            rate_limit_rejections=0,
            streaming_events_emitted=10,
            streaming_errors=0,
            total_duration_ms=500.0,
        )
        data = json.loads(snap.to_json())
        assert data["requests_started"] == 3
        assert data["requests_completed"] == 2
