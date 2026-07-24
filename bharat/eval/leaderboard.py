from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeaderboardEntry:
    checkpoint_name: str
    benchmark_id: str
    category: str
    metric_values: Mapping[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_name:
            raise ValueError("checkpoint_name must be a non-empty string")
        if not self.benchmark_id:
            raise ValueError("benchmark_id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "benchmark_id": self.benchmark_id,
            "category": self.category,
            "metric_values": dict(self.metric_values),
            "aggregate_score": self.aggregate_score,
            "metadata": dict(self.metadata),
        }


class Leaderboard:
    def __init__(self) -> None:
        self._entries: list[LeaderboardEntry] = []

    def add_entry(self, entry: LeaderboardEntry) -> None:
        self._entries.append(entry)

    def rank(
        self,
        benchmark_id: str | None = None,
        category: str | None = None,
    ) -> Sequence[LeaderboardEntry]:
        filtered: list[LeaderboardEntry] = list(self._entries)
        if benchmark_id is not None:
            filtered = [e for e in filtered if e.benchmark_id == benchmark_id]
        if category is not None:
            filtered = [e for e in filtered if e.category == category]
        return tuple(
            sorted(
                filtered,
                key=lambda e: (-e.aggregate_score, e.checkpoint_name),
            )
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def to_json(self, indent: int = 2) -> str:
        d: dict[str, Any] = {
            "leaderboard": [e.to_dict() for e in self.rank()],
            "entry_count": self.entry_count,
        }
        return json.dumps(d, indent=indent)

    def to_markdown(
        self,
        benchmark_id: str | None = None,
        category: str | None = None,
    ) -> str:
        rows = self.rank(benchmark_id=benchmark_id, category=category)
        if not rows:
            return "*(no entries)*\n"

        lines: list[str] = []
        lines.append("| Rank | Checkpoint | Benchmark | Category | Aggregate Score |")
        lines.append("|------|------------|----------|----------|-----------------|")
        for rank, entry in enumerate(rows, 1):
            lines.append(
                f"| {rank} | {entry.checkpoint_name} "
                f"| {entry.benchmark_id} "
                f"| {entry.category} "
                f"| {entry.aggregate_score:.4f} |"
            )
        lines.append("")
        return "\n".join(lines)


def _compute_aggregate_score(metric_values: Mapping[str, float]) -> float:
    if not metric_values:
        return 0.0
    return sum(metric_values.values()) / len(metric_values)


def load_report(path: Path) -> LeaderboardEntry:
    data = json.loads(path.read_text())

    checkpoint_name = data.get("checkpoint_name")
    benchmark_id = data.get("benchmark_id")

    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError(f"Invalid or missing checkpoint_name in {path}")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError(f"Invalid or missing benchmark_id in {path}")

    aggregate_scores_raw = data.get("aggregate_scores", {})
    if not isinstance(aggregate_scores_raw, dict):
        raise ValueError(f"aggregate_scores must be a dict in {path}")

    metric_values: dict[str, float] = {}
    for key, value in aggregate_scores_raw.items():
        if isinstance(value, int | float):
            metric_values[key] = float(value)

    aggregate_score = _compute_aggregate_score(metric_values)

    category_raw = data.get("category", "")
    category: str = category_raw if isinstance(category_raw, str) else ""

    metadata_raw = data.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(metadata_raw, dict):
        for k, v in metadata_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                metadata[k] = v

    return LeaderboardEntry(
        checkpoint_name=checkpoint_name,
        benchmark_id=benchmark_id,
        category=category,
        metric_values=metric_values,
        aggregate_score=aggregate_score,
        metadata=metadata,
    )


def load_leaderboard(reports_root: Path) -> Leaderboard:
    leaderboard = Leaderboard()
    for child in sorted(reports_root.iterdir()):
        if child.suffix == ".json" and not child.name.startswith("."):
            entry = load_report(child)
            leaderboard.add_entry(entry)
    return leaderboard
