from __future__ import annotations

import json
from pathlib import Path

from bharat.reasoning.cot_engine import (
    SOVEREIGN_REASONING_PROBLEMS,
    CoTReasoningEngine,
    ReasoningTraceSample,
)
from scripts.generate_reasoning_curriculum import main as reasoning_data_main
from scripts.generate_reasoning_curriculum import parse_args


class TestCoTReasoningEngine:
    def test_sovereign_reasoning_problems_registry(self):
        assert len(SOVEREIGN_REASONING_PROBLEMS) >= 4
        ids = {p["id"] for p in SOVEREIGN_REASONING_PROBLEMS}
        assert "reasoning_math_ramanujan_01" in ids
        assert "reasoning_physics_orbital_02" in ids

    def test_reasoning_trace_sample_formatting(self):
        engine = CoTReasoningEngine()
        samples = engine.get_samples()
        assert len(samples) > 0

        s = samples[0]
        assert isinstance(s, ReasoningTraceSample)
        formatted = s.to_formatted_cot_text()
        assert "<think>" in formatted
        assert "</think>" in formatted
        assert "<answer>" in formatted
        assert "</answer>" in formatted

        record = s.to_sft_record()
        assert "prompt" in record
        assert "response" in record
        assert "metadata" in record

    def test_export_curriculum(self, tmp_path: Path):
        engine = CoTReasoningEngine()
        out_f = engine.export_curriculum(output_dir=tmp_path / "reasoning_data", multiplier=2)

        assert out_f.is_file()
        with open(out_f, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) >= 8

    def test_cli_parse_args(self):
        args = parse_args(["--multiplier", "3", "--output-dir", "custom/reasoning"])
        assert args.multiplier == 3
        assert args.output_dir == "custom/reasoning"

    def test_cli_main(self, tmp_path: Path):
        code = reasoning_data_main(
            ["--output-dir", str(tmp_path / "cli_reasoning"), "--multiplier", "1"]
        )
        assert code == 0
