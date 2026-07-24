from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.eval.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    _compute_aggregate_score,
    load_leaderboard,
    load_report,
)


class TestLeaderboardEntry:
    def test_valid_minimal(self) -> None:
        entry = LeaderboardEntry(
            checkpoint_name="bharat-350m",
            benchmark_id="language_qa",
            category="language",
        )
        assert entry.checkpoint_name == "bharat-350m"
        assert entry.benchmark_id == "language_qa"
        assert entry.category == "language"
        assert entry.aggregate_score == 0.0

    def test_valid_full(self) -> None:
        entry = LeaderboardEntry(
            checkpoint_name="bharat-350m",
            benchmark_id="language_qa",
            category="language",
            metric_values={"exact_match": 1.0, "token_f1": 0.8},
            aggregate_score=0.9,
            metadata={"params": "350M"},
        )
        assert entry.aggregate_score == 0.9
        assert entry.metadata["params"] == "350M"

    def test_empty_checkpoint_name_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_name"):
            LeaderboardEntry(
                checkpoint_name="",
                benchmark_id="language_qa",
                category="language",
            )

    def test_empty_benchmark_id_raises(self) -> None:
        with pytest.raises(ValueError, match="benchmark_id"):
            LeaderboardEntry(
                checkpoint_name="bharat-350m",
                benchmark_id="",
                category="language",
            )

    def test_to_dict(self) -> None:
        entry = LeaderboardEntry(
            checkpoint_name="bharat-350m",
            benchmark_id="language_qa",
            category="language",
            metric_values={"exact_match": 1.0},
            aggregate_score=1.0,
            metadata={"params": "350M"},
        )
        d = entry.to_dict()
        assert d["checkpoint_name"] == "bharat-350m"
        assert d["aggregate_score"] == 1.0
        assert d["metric_values"]["exact_match"] == 1.0


class TestLeaderboard:
    def test_empty_leaderboard(self) -> None:
        lb = Leaderboard()
        assert lb.entry_count == 0
        assert lb.rank() == ()

    def test_add_entry(self) -> None:
        lb = Leaderboard()
        entry = LeaderboardEntry(
            checkpoint_name="bharat-350m",
            benchmark_id="language_qa",
            category="language",
            aggregate_score=0.9,
        )
        lb.add_entry(entry)
        assert lb.entry_count == 1

    def test_rank_sorts_by_score_descending(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-c",
                benchmark_id="language_qa",
                category="language",
                aggregate_score=0.5,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="language_qa",
                category="language",
                aggregate_score=0.9,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-b",
                benchmark_id="language_qa",
                category="language",
                aggregate_score=0.7,
            )
        )
        ranked = lb.rank()
        assert [e.checkpoint_name for e in ranked] == ["model-a", "model-b", "model-c"]
        assert [e.aggregate_score for e in ranked] == [0.9, 0.7, 0.5]

    def test_tie_breaking_by_checkpoint_name(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-b",
                benchmark_id="language_qa",
                category="language",
                aggregate_score=0.5,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="language_qa",
                category="language",
                aggregate_score=0.5,
            )
        )
        ranked = lb.rank()
        assert ranked[0].checkpoint_name == "model-a"
        assert ranked[1].checkpoint_name == "model-b"

    def test_filter_by_benchmark_id(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_2",
                category="reasoning",
                aggregate_score=0.8,
            )
        )
        filtered = lb.rank(benchmark_id="bench_1")
        assert len(filtered) == 1
        assert filtered[0].benchmark_id == "bench_1"

    def test_filter_by_category(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_2",
                category="reasoning",
                aggregate_score=0.8,
            )
        )
        filtered = lb.rank(category="reasoning")
        assert len(filtered) == 1
        assert filtered[0].category == "reasoning"

    def test_filter_by_both(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="reasoning",
                aggregate_score=0.8,
            )
        )
        filtered = lb.rank(benchmark_id="bench_1", category="language")
        assert len(filtered) == 1
        assert filtered[0].category == "language"

    def test_to_json(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        output = lb.to_json()
        data = json.loads(output)
        assert data["entry_count"] == 1
        assert len(data["leaderboard"]) == 1
        assert data["leaderboard"][0]["checkpoint_name"] == "model-a"

    def test_to_markdown(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        md = lb.to_markdown()
        assert "| Rank | Checkpoint | Benchmark | Category | Aggregate Score |" in md
        assert "| 1 | model-a | bench_1 | language | 0.9000 |" in md

    def test_to_markdown_empty(self) -> None:
        lb = Leaderboard()
        md = lb.to_markdown()
        assert md == "*(no entries)*\n"

    def test_to_markdown_with_filter(self) -> None:
        lb = Leaderboard()
        lb.add_entry(
            LeaderboardEntry(
                checkpoint_name="model-a",
                benchmark_id="bench_1",
                category="language",
                aggregate_score=0.9,
            )
        )
        md = lb.to_markdown(benchmark_id="nonexistent")
        assert md == "*(no entries)*\n"


class TestComputeAggregateScore:
    def test_empty_returns_zero(self) -> None:
        assert _compute_aggregate_score({}) == 0.0

    def test_single_value(self) -> None:
        assert _compute_aggregate_score({"exact_match": 1.0}) == 1.0

    def test_multiple_values(self) -> None:
        score = _compute_aggregate_score({"a": 1.0, "b": 0.5, "c": 0.0})
        assert score == 0.5


class TestLoadReport:
    def test_valid_report(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint_name": "bharat-350m",
                    "benchmark_id": "language_qa",
                    "category": "language",
                    "aggregate_scores": {"overall_exact_match": 0.8, "overall_token_f1": 0.6},
                    "metadata": {"params": "350M"},
                }
            )
        )
        entry = load_report(report_path)
        assert entry.checkpoint_name == "bharat-350m"
        assert entry.benchmark_id == "language_qa"
        assert entry.category == "language"
        assert entry.aggregate_score == 0.7

    def test_missing_checkpoint_name_raises(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "benchmark_id": "language_qa",
                }
            )
        )
        with pytest.raises(ValueError, match="checkpoint_name"):
            load_report(report_path)

    def test_missing_benchmark_id_raises(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint_name": "bharat-350m",
                }
            )
        )
        with pytest.raises(ValueError, match="benchmark_id"):
            load_report(report_path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            load_report(report_path)

    def test_non_dict_aggregate_scores(self, tmp_path: Path) -> None:
        report_path = tmp_path / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "checkpoint_name": "bharat-350m",
                    "benchmark_id": "language_qa",
                    "aggregate_scores": "invalid",
                }
            )
        )
        with pytest.raises(ValueError, match="aggregate_scores must be a dict"):
            load_report(report_path)


class TestLoadLeaderboard:
    def test_loads_multiple_reports(self) -> None:
        fixtures_root = Path("eval_fixtures/leaderboard")
        if not fixtures_root.exists():
            pytest.skip("Leaderboard fixtures not found")
        lb = load_leaderboard(fixtures_root)
        assert lb.entry_count >= 5

    def test_ranking_from_fixtures(self) -> None:
        fixtures_root = Path("eval_fixtures/leaderboard")
        if not fixtures_root.exists():
            pytest.skip("Leaderboard fixtures not found")
        lb = load_leaderboard(fixtures_root)
        ranked = lb.rank(benchmark_id="language_qa")
        assert len(ranked) >= 3
        scores = [e.aggregate_score for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_directory(self, tmp_path: Path) -> None:
        lb = load_leaderboard(tmp_path)
        assert lb.entry_count == 0

    def test_ignores_non_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        lb = load_leaderboard(tmp_path)
        assert lb.entry_count == 0
