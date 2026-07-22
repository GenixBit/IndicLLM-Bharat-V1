from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bharat.eval.schema import EvalExample


class PredictionAdapter(Protocol):
    def predict(self, example: EvalExample) -> str:
        ...


@dataclass(frozen=True)
class EchoPredictionAdapter:
    def predict(self, example: EvalExample) -> str:
        return example.prompt


@dataclass(frozen=True)
class ExpectedPredictionAdapter:
    """Smoke-test adapter that returns expected answers.

    This adapter is only for deterministic local test flows. It must not be
    used as a real model baseline.
    """

    def predict(self, example: EvalExample) -> str:
        return example.expected


@dataclass(frozen=True)
class ChoiceBaselineAdapter:
    def predict(self, example: EvalExample) -> str:
        if example.task_type != "classification":
            raise ValueError(
                "ChoiceBaselineAdapter only supports classification examples, "
                f"got {example.task_type!r}"
            )
        if not example.choices:
            raise ValueError("classification examples must include choices")
        return example.choices[0]


def build_prediction_adapter(name: str) -> PredictionAdapter:
    if name == "expected":
        return ExpectedPredictionAdapter()
    if name == "echo":
        return EchoPredictionAdapter()
    if name == "choice-baseline":
        return ChoiceBaselineAdapter()
    raise ValueError(
        "adapter must be one of ['choice-baseline', 'echo', 'expected'], "
        f"got {name!r}"
    )
