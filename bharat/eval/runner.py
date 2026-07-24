from __future__ import annotations

from collections.abc import Sequence

from bharat.eval.metrics import (
    choice_accuracy,
    exact_match,
    normalized_exact_match,
    token_f1,
)
from bharat.eval.schema import EvalExample, EvalPrediction, EvalResult


class BharatBenchRunner:
    def run(
        self,
        examples: Sequence[EvalExample],
        predictions: Sequence[EvalPrediction],
    ) -> tuple[EvalResult, ...]:
        pred_by_id: dict[str, EvalPrediction] = {}
        for p in predictions:
            if p.example_id in pred_by_id:
                raise ValueError(f"Duplicate prediction for example_id {p.example_id!r}")
            pred_by_id[p.example_id] = p

        results: list[EvalResult] = []
        seen: set[str] = set()

        for ex in sorted(examples, key=lambda e: e.example_id):
            if ex.example_id in seen:
                raise ValueError(f"Duplicate example_id {ex.example_id!r}")
            seen.add(ex.example_id)

            if ex.example_id not in pred_by_id:
                raise ValueError(f"Missing prediction for example_id {ex.example_id!r}")

            pred = pred_by_id[ex.example_id]

            scores = self._compute_scores(ex, pred.prediction)
            results.append(
                EvalResult(
                    example_id=ex.example_id,
                    task_type=ex.task_type,
                    expected=ex.expected,
                    prediction=pred.prediction,
                    scores=scores,
                )
            )

        used_ids = {p.example_id for p in predictions}
        example_ids = {e.example_id for e in examples}
        unknown = used_ids - example_ids
        if unknown:
            raise ValueError(f"Unknown prediction IDs: {sorted(unknown)}")

        return tuple(results)

    @staticmethod
    def _compute_scores(example: EvalExample, prediction: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        if example.task_type == "qa":
            scores["exact_match"] = exact_match(example.expected, prediction)
            scores["normalized_exact_match"] = normalized_exact_match(example.expected, prediction)
            scores["token_f1"] = token_f1(example.expected, prediction)
        elif example.task_type == "classification":
            scores["choice_accuracy"] = choice_accuracy(
                example.expected, prediction, example.choices
            )
        elif example.task_type == "generation":
            scores["normalized_exact_match"] = normalized_exact_match(example.expected, prediction)
            scores["token_f1"] = token_f1(example.expected, prediction)
        return scores
