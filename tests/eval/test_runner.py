from __future__ import annotations

import pytest

from bharat.eval.runner import BharatBenchRunner
from bharat.eval.schema import EvalExample, EvalPrediction


def _make_examples() -> list[EvalExample]:
    return [
        EvalExample(example_id="qa_001", task_type="qa", prompt="Q1?", expected="A"),
        EvalExample(
            example_id="cls_001",
            task_type="classification",
            prompt="Q2?",
            expected="A",
            choices=("A", "B"),
        ),
        EvalExample(
            example_id="gen_001", task_type="generation", prompt="Q3?", expected="hello world"
        ),
    ]


def _make_predictions() -> list[EvalPrediction]:
    return [
        EvalPrediction(example_id="qa_001", prediction="A"),
        EvalPrediction(example_id="cls_001", prediction="A"),
        EvalPrediction(example_id="gen_001", prediction="hello world"),
    ]


class TestBharatBenchRunner:
    def test_run_succeeds(self) -> None:
        runner = BharatBenchRunner()
        examples = _make_examples()
        predictions = _make_predictions()
        results = runner.run(examples, predictions)
        assert len(results) == 3
        assert results[0].example_id == "cls_001"
        assert results[1].example_id == "gen_001"
        assert results[2].example_id == "qa_001"

    def test_missing_prediction_fails(self) -> None:
        runner = BharatBenchRunner()
        exs = [EvalExample(example_id="ex_001", task_type="qa", prompt="Q?", expected="A")]
        preds: list[EvalPrediction] = []
        with pytest.raises(ValueError, match="Missing prediction"):
            runner.run(exs, preds)

    def test_duplicate_prediction_fails(self) -> None:
        runner = BharatBenchRunner()
        exs = [EvalExample(example_id="ex_001", task_type="qa", prompt="Q?", expected="A")]
        preds = [
            EvalPrediction(example_id="ex_001", prediction="A"),
            EvalPrediction(example_id="ex_001", prediction="B"),
        ]
        with pytest.raises(ValueError, match="Duplicate prediction"):
            runner.run(exs, preds)

    def test_unknown_prediction_fails(self) -> None:
        runner = BharatBenchRunner()
        exs = [EvalExample(example_id="ex_001", task_type="qa", prompt="Q?", expected="A")]
        preds = [
            EvalPrediction(example_id="ex_001", prediction="A"),
            EvalPrediction(example_id="unknown", prediction="B"),
        ]
        with pytest.raises(ValueError, match="Unknown prediction"):
            runner.run(exs, preds)

    def test_deterministic_order(self) -> None:
        runner = BharatBenchRunner()
        examples = _make_examples()
        predictions = _make_predictions()
        results1 = runner.run(examples, predictions)
        results2 = runner.run(examples, predictions)
        assert [r.example_id for r in results1] == [r.example_id for r in results2]

    def test_qa_metrics(self) -> None:
        runner = BharatBenchRunner()
        examples = [
            EvalExample(example_id="qa_001", task_type="qa", prompt="Q?", expected="New Delhi")
        ]
        predictions = [EvalPrediction(example_id="qa_001", prediction="New Delhi")]
        results = runner.run(examples, predictions)
        assert results[0].scores["exact_match"] == 1.0
        assert results[0].scores["normalized_exact_match"] == 1.0
        assert results[0].scores["token_f1"] == 1.0

    def test_classification_metrics(self) -> None:
        runner = BharatBenchRunner()
        examples = [
            EvalExample(
                example_id="cls_001",
                task_type="classification",
                prompt="Q?",
                expected="A",
                choices=("A", "B"),
            )
        ]
        predictions = [EvalPrediction(example_id="cls_001", prediction="A")]
        results = runner.run(examples, predictions)
        assert results[0].scores["choice_accuracy"] == 1.0

    def test_generation_metrics(self) -> None:
        runner = BharatBenchRunner()
        examples = [
            EvalExample(
                example_id="gen_001", task_type="generation", prompt="Q?", expected="hello world"
            )
        ]
        predictions = [EvalPrediction(example_id="gen_001", prediction="hello world")]
        results = runner.run(examples, predictions)
        assert results[0].scores["normalized_exact_match"] == 1.0
        assert results[0].scores["token_f1"] == 1.0
