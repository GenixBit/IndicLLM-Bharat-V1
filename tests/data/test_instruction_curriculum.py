from __future__ import annotations

import json
from pathlib import Path

from bharat.data.instruction_curriculum import (
    INDIC_INSTRUCTION_DATA,
    STEM_INSTRUCTION_DATA,
    export_instruction_curriculum,
    get_all_instruction_curriculum,
)
from scripts.generate_instruction_curriculum import main as generate_sft_main
from scripts.generate_instruction_curriculum import parse_args


class TestInstructionCurriculum:
    def test_curriculum_coverage(self):
        items = get_all_instruction_curriculum()
        assert len(items) >= len(STEM_INSTRUCTION_DATA) + len(INDIC_INSTRUCTION_DATA)
        for item in items:
            assert "prompt" in item and len(item["prompt"]) > 5
            assert "response" in item and len(item["response"]) > 20

    def test_export_curriculum(self, tmp_path: Path):
        out_p = tmp_path / "test_curriculum.jsonl"
        count = export_instruction_curriculum(out_p)
        assert count > 0
        assert out_p.is_file()

        lines = out_p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == count
        first = json.loads(lines[0])
        assert "prompt" in first
        assert "response" in first

    def test_cli_parse_args(self):
        args = parse_args(["--output", "custom_sft.jsonl"])
        assert args.output == "custom_sft.jsonl"

    def test_cli_main(self, tmp_path: Path):
        out_p = tmp_path / "cli_sft.jsonl"
        code = generate_sft_main(["--output", str(out_p)])
        assert code == 0
        assert out_p.is_file()
