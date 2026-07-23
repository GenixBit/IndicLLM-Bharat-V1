from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from bharat.eval.adapters import PredictionAdapter
from bharat.eval.schema import EvalExample, EvalPrediction

_URL_RE = re.compile(r"^(https?|ftp|s3|gs):/+", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


class PredictionRunner:
    def run(
        self,
        examples: Sequence[EvalExample],
        adapter: PredictionAdapter,
    ) -> tuple[EvalPrediction, ...]:
        seen_ids: set[str] = set()
        predictions: list[EvalPrediction] = []

        for example in sorted(examples, key=lambda e: e.example_id):
            if example.example_id in seen_ids:
                raise ValueError(f"Duplicate example_id {example.example_id!r}")
            seen_ids.add(example.example_id)

            prediction = adapter.predict(example)
            if not isinstance(prediction, str):
                raise TypeError(
                    "Adapter returned non-string prediction for "
                    f"{example.example_id!r}: {type(prediction).__name__}"
                )
            predictions.append(EvalPrediction(example_id=example.example_id, prediction=prediction))

        return tuple(predictions)


def write_predictions_jsonl(
    predictions: Sequence[EvalPrediction],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    if _is_remote_url(str(output)):
        raise ValueError(f"Remote output path rejected: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(p.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for p in predictions
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
