from __future__ import annotations

from bharat.eval.reporting import BharatBenchReport, compute_aggregate_scores
from bharat.eval.schema import EvalResult


def _make_results() -> tuple[EvalResult, ...]:
    return (
        EvalResult(
            example_id="qa_001",
            task_type="qa",
            expected="A",
            prediction="A",
            scores={
                "exact_match": 1.0,
                "normalized_exact_match": 1.0,
                "token_f1": 1.0,
            },
        ),
        EvalResult(
            example_id="qa_002",
            task_type="qa",
            expected="A",
            prediction="B",
            scores={
                "exact_match": 0.0,
                "normalized_exact_match": 0.0,
                "token_f1": 0.0,
            },
        ),
        EvalResult(
            example_id="cls_001",
            task_type="classification",
            expected="A",
            prediction="A",
            scores={"choice_accuracy": 1.0},
        ),
    )


class TestBharatBenchReport:
    def test_minimal_valid(self) -> None:
        report = BharatBenchReport(
            run_id="run-001",
            example_count=3,
            task_counts={"qa": 2, "classification": 1},
            aggregate_scores={"overall_exact_match": 0.5},
            results=(),
            created_at="2026-07-20T00:00:00Z",
        )
        assert report.run_id == "run-001"
        assert report.digest()

    def test_digest_deterministic(self) -> None:
        r1 = BharatBenchReport(
            run_id="run-001",
            example_count=1,
            task_counts={"qa": 1},
            aggregate_scores={"overall_exact_match": 1.0},
            results=(),
            created_at="2026-07-20T00:00:00Z",
        )
        r2 = BharatBenchReport(
            run_id="run-001",
            example_count=1,
            task_counts={"qa": 1},
            aggregate_scores={"overall_exact_match": 1.0},
            results=(),
            created_at="2026-07-20T00:00:00Z",
        )
        assert r1.digest() == r2.digest()

    def test_aggregate_metrics_deterministic(self) -> None:
        results = _make_results()
        agg1 = compute_aggregate_scores(results)
        agg2 = compute_aggregate_scores(results)
        assert agg1 == agg2

    def test_report_to_dict(self) -> None:
        report = BharatBenchReport(
            run_id="run-001",
            example_count=3,
            task_counts={"qa": 2, "classification": 1},
            aggregate_scores={"overall_exact_match": 0.5},
            results=(),
            created_at="2026-07-20T00:00:00Z",
        )
        d = report.to_dict()
        assert d["run_id"] == "run-001"
        assert d["example_count"] == 3
