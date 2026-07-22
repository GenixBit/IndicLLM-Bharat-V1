from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bharat.eval.schema import EvalResult

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class BharatBenchReport:
    run_id: str
    example_count: int
    task_counts: Mapping[str, int]
    aggregate_scores: Mapping[str, float]
    results: tuple[EvalResult, ...]
    created_at: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if self.example_count < 0:
            raise ValueError(
                f"example_count must be non-negative, got {self.example_count}"
            )
        if not _ISO_UTC_RE.match(self.created_at):
            raise ValueError(
                f"created_at must be ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ), "
                f"got {self.created_at!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "example_count": self.example_count,
            "task_counts": dict(self.task_counts),
            "aggregate_scores": dict(self.aggregate_scores),
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at,
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_aggregate_scores(results: Sequence[EvalResult]) -> dict[str, float]:
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in results:
        by_task[r.task_type].append(dict(r.scores))

    aggregates: dict[str, float] = {}
    all_scores: dict[str, list[float]] = defaultdict(list)

    for task_type, task_results in by_task.items():
        for metric_name in task_results[0]:
            values = [r[metric_name] for r in task_results]
            mean = sum(values) / len(values)
            key = f"{task_type}_{metric_name}"
            aggregates[key] = mean
            all_scores[metric_name].extend(values)

    for metric_name, values in all_scores.items():
        key = f"overall_{metric_name}"
        aggregates[key] = sum(values) / len(values)

    return aggregates
