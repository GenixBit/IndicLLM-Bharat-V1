from __future__ import annotations

import pytest

from bharat.eval.adapters import (
    ChoiceBaselineAdapter,
    EchoPredictionAdapter,
    ExpectedPredictionAdapter,
    build_prediction_adapter,
)
from bharat.eval.schema import EvalExample


def test_expected_prediction_adapter_returns_expected() -> None:
    ex = EvalExample(
        example_id="qa_001",
        task_type="qa",
        prompt="Question?",
        expected="Answer",
    )
    assert ExpectedPredictionAdapter().predict(ex) == "Answer"


def test_echo_prediction_adapter_returns_prompt() -> None:
    ex = EvalExample(
        example_id="qa_001",
        task_type="qa",
        prompt="Question?",
        expected="Answer",
    )
    assert EchoPredictionAdapter().predict(ex) == "Question?"


def test_choice_baseline_adapter_returns_first_choice() -> None:
    ex = EvalExample(
        example_id="cls_001",
        task_type="classification",
        prompt="Language?",
        expected="Hindi",
        choices=("Hindi", "Marathi"),
    )
    assert ChoiceBaselineAdapter().predict(ex) == "Hindi"


def test_choice_baseline_adapter_rejects_non_classification() -> None:
    ex = EvalExample(
        example_id="qa_001",
        task_type="qa",
        prompt="Question?",
        expected="Answer",
    )
    with pytest.raises(ValueError, match="only supports classification"):
        ChoiceBaselineAdapter().predict(ex)


def test_build_prediction_adapter_known_names() -> None:
    assert isinstance(build_prediction_adapter("expected"), ExpectedPredictionAdapter)
    assert isinstance(build_prediction_adapter("echo"), EchoPredictionAdapter)
    assert isinstance(build_prediction_adapter("choice-baseline"), ChoiceBaselineAdapter)


def test_build_prediction_adapter_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="adapter must be one of"):
        build_prediction_adapter("remote-model")
