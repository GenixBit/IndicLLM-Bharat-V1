from __future__ import annotations

import json
from typing import Any

import pytest

from bharat.eval.adapters import ExpectedPredictionAdapter
from bharat.eval.prediction_runner import PredictionRunner, write_predictions_jsonl
from bharat.eval.schema import EvalExample


def _example(example_id: str, expected: str = "A") -> EvalExample:
    return EvalExample(
        example_id=example_id,
        task_type="qa",
        prompt=f"Prompt {example_id}",
        expected=expected,
    )


def test_prediction_runner_emits_one_prediction_per_example() -> None:
    examples = [_example("b", "B"), _example("a", "A")]
    predictions = PredictionRunner().run(examples, ExpectedPredictionAdapter())
    assert len(predictions) == 2
    assert {p.example_id for p in predictions} == {"a", "b"}


def test_prediction_runner_sorts_deterministically_by_example_id() -> None:
    examples = [_example("b", "B"), _example("a", "A")]
    predictions = PredictionRunner().run(examples, ExpectedPredictionAdapter())
    assert [p.example_id for p in predictions] == ["a", "b"]


def test_prediction_runner_rejects_duplicate_example_ids() -> None:
    examples = [_example("a", "A1"), _example("a", "A2")]
    with pytest.raises(ValueError, match="Duplicate example_id"):
        PredictionRunner().run(examples, ExpectedPredictionAdapter())


def test_prediction_runner_rejects_non_string_adapter_output() -> None:
    class BadAdapter:
        def predict(self, example: EvalExample) -> Any:
            return 123

    with pytest.raises(TypeError, match="non-string prediction"):
        PredictionRunner().run([_example("a")], BadAdapter())


def test_write_predictions_jsonl_writes_deterministic_jsonl(tmp_path) -> None:
    predictions = PredictionRunner().run(
        [_example("b", "B"), _example("a", "A")], ExpectedPredictionAdapter()
    )
    out = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(predictions, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        '{"example_id":"a","prediction":"A"}',
        '{"example_id":"b","prediction":"B"}',
    ]
    assert [json.loads(line) for line in lines] == [
        {"example_id": "a", "prediction": "A"},
        {"example_id": "b", "prediction": "B"},
    ]


def test_write_predictions_jsonl_rejects_remote_path() -> None:
    predictions = PredictionRunner().run([_example("a")], ExpectedPredictionAdapter())
    with pytest.raises(ValueError, match="Remote output path rejected"):
        write_predictions_jsonl(predictions, "https://example.com/predictions.jsonl")
