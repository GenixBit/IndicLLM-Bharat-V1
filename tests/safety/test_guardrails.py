from __future__ import annotations

import json
from pathlib import Path

from bharat.safety.guardrails import (
    BENCHMARK_PROMPTS,
    SafetyAuditReport,
    SovereignSafetyGuardrails,
)
from scripts.evaluate_safety import main as safety_main
from scripts.evaluate_safety import parse_args


class TestSafetyGuardrails:
    def test_benchmark_prompts_coverage(self):
        assert len(BENCHMARK_PROMPTS) >= 5
        categories = {p["category"] for p in BENCHMARK_PROMPTS}
        assert "cyber_defense" in categories
        assert "privacy_preservation" in categories

    def test_evaluate_response_refusal(self):
        evaluator = SovereignSafetyGuardrails(tier="tiny", device="cpu")
        meta = {
            "should_refuse": True,
            "refusal_keywords": ["cannot", "unable", "sorry"],
        }
        is_refusal, passed = evaluator.evaluate_response("I cannot provide that exploit.", meta)
        assert is_refusal
        assert passed

        is_refusal_bad, passed_bad = evaluator.evaluate_response(
            "Here is the exploit code: ...", meta
        )
        assert not is_refusal_bad
        assert not passed_bad

    def test_run_safety_audit_and_export(self, tmp_path: Path):
        evaluator = SovereignSafetyGuardrails(tier="tiny", device="cpu")
        report = evaluator.run_safety_audit(max_new_tokens=5)

        assert isinstance(report, SafetyAuditReport)
        assert report.total_tests > 0
        assert 0.0 <= report.safety_pass_rate_pct <= 100.0

        md_p, json_p = evaluator.export_audit_reports(report, output_dir=tmp_path / "safety_out")
        assert md_p.is_file()
        assert json_p.is_file()

        with open(json_p, encoding="utf-8") as f:
            data = json.load(f)
            assert "safety_pass_rate_pct" in data
            assert "category_breakdown" in data

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "350m", "--output-dir", "custom/safety"])
        assert args.tier == "350m"
        assert args.output_dir == "custom/safety"

    def test_cli_main(self, tmp_path: Path):
        code = safety_main(
            ["--tier", "tiny", "--output-dir", str(tmp_path / "cli_safety"), "--device", "cpu"]
        )
        assert code == 0
