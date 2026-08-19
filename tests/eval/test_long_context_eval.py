from __future__ import annotations

from pathlib import Path

from bharat.eval.long_context import LongContextEvaluator
from scripts.evaluate_long_context import main as eval_long_main
from scripts.evaluate_long_context import parse_args


class TestLongContextEvaluator:
    def test_evaluator_grid(self):
        evaluator = LongContextEvaluator(tier="tiny", device="cpu")
        report = evaluator.run_benchmark(
            context_lengths=[256, 512],
            depths=[10, 50],
        )

        assert report.overall_accuracy_pct >= 0.0
        assert len(report.results) == 2 * 2 * 4  # 2 lengths * 2 depths * 4 languages
        assert "IndicLLM-Bharat Long-Context Evaluation" in report.summary_markdown

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "1b", "--context-lengths", "4096", "8192"])
        assert args.tier == "1b"
        assert args.context_lengths == [4096, 8192]

    def test_cli_main(self, tmp_path: Path):
        out_report = tmp_path / "long_report.md"
        code = eval_long_main(
            [
                "--tier",
                "tiny",
                "--context-lengths",
                "128",
                "--depths",
                "50",
                "--device",
                "cpu",
                "--output-report",
                str(out_report),
            ]
        )
        assert code == 0
        assert out_report.is_file()
