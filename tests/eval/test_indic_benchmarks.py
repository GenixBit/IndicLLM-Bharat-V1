from __future__ import annotations

from pathlib import Path

from bharat.eval.indic_benchmarks import (
    INDIC_MMLU_TASKS,
    IndicBenchmarkRunner,
)
from scripts.evaluate_indic_benchmarks import main as benchmark_main
from scripts.evaluate_indic_benchmarks import parse_args


class TestIndicBenchmarks:
    def test_benchmark_tasks_structure(self):
        assert len(INDIC_MMLU_TASKS) >= 12
        for task in INDIC_MMLU_TASKS:
            assert "id" in task
            assert "lang" in task
            assert "subject" in task
            assert "question" in task
            assert "options" in task
            assert set(task["options"].keys()) == {"A", "B", "C", "D"}
            assert task["answer"] in {"A", "B", "C", "D"}

    def test_benchmark_runner_evaluates(self, tmp_path: Path):
        # Runner fallback loads dummy model when checkpoint not found
        runner = IndicBenchmarkRunner(
            checkpoint_path=tmp_path / "dummy.pt",
            tokenizer_name="gpt2",
            device="cpu",
        )

        res = runner.evaluate_mmlu(INDIC_MMLU_TASKS[:4])
        assert res.total_questions == 4
        assert 0 <= res.correct_answers <= 4
        assert 0.0 <= res.accuracy_pct <= 100.0
        assert len(res.per_language_accuracy) > 0

    def test_report_generation(self, tmp_path: Path):
        runner = IndicBenchmarkRunner(
            checkpoint_path=tmp_path / "dummy.pt",
            tokenizer_name="gpt2",
            device="cpu",
        )
        report = runner.generate_report()
        assert report.model_name == "IndicLLM-Bharat"
        assert "IndicLLM-Bharat Benchmark Evaluation Report" in report.summary_markdown
        d = report.to_dict()
        assert "mmlu_metrics" in d

    def test_cli_parse_args(self):
        args = parse_args(
            [
                "--checkpoint",
                "custom.pt",
                "--device",
                "cpu",
                "--output-report",
                "rep.md",
            ]
        )
        assert args.checkpoint == "custom.pt"
        assert args.device == "cpu"
        assert args.output_report == "rep.md"

    def test_cli_main(self, tmp_path: Path):
        out_report = tmp_path / "test_report.md"
        out_json = tmp_path / "test_metrics.json"

        code = benchmark_main(
            [
                "--checkpoint",
                str(tmp_path / "nonexistent.pt"),
                "--device",
                "cpu",
                "--output-report",
                str(out_report),
                "--json-output",
                str(out_json),
            ]
        )
        assert code == 0
        assert out_report.is_file()
        assert out_json.is_file()
