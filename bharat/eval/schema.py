from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

_SUPPORTED_TASK_TYPES = frozenset({"qa", "classification", "generation"})


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    task_type: str
    prompt: str
    expected: str
    choices: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be a non-empty string")
        if not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.task_type not in _SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type {self.task_type!r}; "
                f"must be one of {sorted(_SUPPORTED_TASK_TYPES)}"
            )
        if self.task_type == "classification" and not self.choices:
            raise ValueError("classification tasks must have at least one choice")

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "expected": self.expected,
            "choices": list(self.choices),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> EvalExample:
        kwargs = dict(d)
        choices = kwargs.pop("choices", ())
        if isinstance(choices, list):
            choices = tuple(choices)
        kwargs["choices"] = choices
        return cls(**kwargs)

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvalPrediction:
    example_id: str
    prediction: str

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"example_id": self.example_id, "prediction": self.prediction}


@dataclass(frozen=True)
class EvalResult:
    example_id: str
    task_type: str
    expected: str
    prediction: str
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be a non-empty string")
        if not self.task_type:
            raise ValueError("task_type must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "task_type": self.task_type,
            "expected": self.expected,
            "prediction": self.prediction,
            "scores": dict(self.scores),
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
