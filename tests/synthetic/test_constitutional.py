from __future__ import annotations

import json
from pathlib import Path

from bharat.synthetic.constitutional import (
    SOVEREIGN_CONSTITUTIONAL_PRINCIPLES,
    ConstitutionalDataEngine,
    ConstitutionalPair,
)
from scripts.generate_constitutional_data import main as const_data_main
from scripts.generate_constitutional_data import parse_args


class TestConstitutionalEngine:
    def test_principles_registry(self):
        assert len(SOVEREIGN_CONSTITUTIONAL_PRINCIPLES) >= 4
        ids = {p["id"] for p in SOVEREIGN_CONSTITUTIONAL_PRINCIPLES}
        assert "const_equality_01" in ids
        assert "const_cyber_safety_03" in ids

    def test_generate_curated_pairs(self):
        engine = ConstitutionalDataEngine()
        pairs = engine.generate_curated_pairs()
        assert len(pairs) > 0
        p = pairs[0]
        assert isinstance(p, ConstitutionalPair)
        assert len(p.prompt) > 0
        assert len(p.chosen) > 0
        assert len(p.rejected) > 0

        dpo_d = p.to_dpo_dict()
        assert "prompt" in dpo_d
        assert "chosen" in dpo_d
        assert "rejected" in dpo_d

        sft_d = p.to_sft_dict()
        assert "prompt" in sft_d
        assert "response" in sft_d

    def test_export_datasets(self, tmp_path: Path):
        engine = ConstitutionalDataEngine()
        dpo_f, sft_f = engine.export_datasets(output_dir=tmp_path / "const_data", num_multiplier=2)

        assert dpo_f.is_file()
        assert sft_f.is_file()

        with open(dpo_f, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) >= 10

    def test_cli_parse_args(self):
        args = parse_args(["--multiplier", "3", "--output-dir", "custom/dir"])
        assert args.multiplier == 3
        assert args.output_dir == "custom/dir"

    def test_cli_main(self, tmp_path: Path):
        code = const_data_main(["--output-dir", str(tmp_path / "cli_const"), "--multiplier", "1"])
        assert code == 0
