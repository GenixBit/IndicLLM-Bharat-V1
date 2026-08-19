from __future__ import annotations

from pathlib import Path

from bharat.eval.scale_evaluator import ScaleTierEvaluator
from scripts.evaluate_scale_tiers import main as eval_scale_main
from scripts.evaluate_scale_tiers import parse_args


class TestScaleEvaluator:
    def test_evaluator_generation(self, tmp_path: Path):
        evaluator = ScaleTierEvaluator(
            tiers=["1b", "3b"],
            checkpoints_base=tmp_path,
            device="cpu",
        )
        report = evaluator.generate_comparison_matrix()
        assert len(report.tiers) == 2
        assert "IndicLLM-Bharat Multi-Tier Scaling Matrix" in report.summary_markdown
        d = report.to_dict()
        assert "tiers" in d

    def test_cli_parse_args(self):
        args = parse_args(["--tiers", "1b", "10b", "--device", "cpu"])
        assert args.tiers == ["1b", "10b"]
        assert args.device == "cpu"

    def test_cli_main(self, tmp_path: Path):
        out_report = tmp_path / "scale_report.md"
        code = eval_scale_main(
            [
                "--tiers",
                "1b",
                "--checkpoints-base",
                str(tmp_path),
                "--device",
                "cpu",
                "--output-report",
                str(out_report),
            ]
        )
        assert code == 0
        assert out_report.is_file()
