from __future__ import annotations

import json
from pathlib import Path

from bharat.reasoning.verifier import (
    ReasoningEvaluationReport,
    ReasoningVerifier,
)
from scripts.evaluate_reasoning import main as reasoning_eval_main
from scripts.evaluate_reasoning import parse_args


class TestReasoningVerifier:
    def test_parse_reasoning_trace(self):
        verifier = ReasoningVerifier(tier="tiny", device="cpu")
        sample_output = (
            "<think>\n1. Step one calculation\n2. Step two deduction\n</think>\n"
            "<answer>\nThe final answer is 42.\n</answer>"
        )
        has_think, has_answer, thought, answer = verifier.parse_reasoning_trace(sample_output)
        assert has_think
        assert has_answer
        assert "Step one calculation" in thought
        assert "The final answer is 42." in answer

        invalid_output = "This is a direct output without tags."
        h_t, h_a, _, _ = verifier.parse_reasoning_trace(invalid_output)
        assert not h_t
        assert not h_a

    def test_evaluate_problems_and_export(self, tmp_path: Path):
        verifier = ReasoningVerifier(tier="tiny", device="cpu")
        report = verifier.evaluate_problems(max_new_tokens=5)

        assert isinstance(report, ReasoningEvaluationReport)
        assert report.total_problems > 0
        assert 0.0 <= report.structure_valid_pct <= 100.0

        md_p, json_p = verifier.export_reports(report, output_dir=tmp_path / "reasoning_out")
        assert md_p.is_file()
        assert json_p.is_file()

        with open(json_p, encoding="utf-8") as f:
            data = json.load(f)
            assert "structure_valid_pct" in data
            assert "per_domain_validity" in data

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "350m", "--output-dir", "custom/reasoning"])
        assert args.tier == "350m"
        assert args.output_dir == "custom/reasoning"

    def test_cli_main(self, tmp_path: Path):
        code = reasoning_eval_main(
            [
                "--tier",
                "tiny",
                "--output-dir",
                str(tmp_path / "cli_reasoning_eval"),
                "--device",
                "cpu",
            ]
        )
        assert code == 0
