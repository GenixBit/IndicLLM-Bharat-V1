from __future__ import annotations

import json
from pathlib import Path

from bharat.data.preference_curriculum import (
    INDIC_PREFERENCES,
    STEM_AND_GLOBAL_PREFERENCES,
    export_preference_curriculum,
    get_all_preference_samples,
)
from scripts.generate_preference_curriculum import main as pref_main
from scripts.generate_preference_curriculum import parse_args


class TestPreferenceCurriculum:
    def test_samples_structure(self):
        samples = get_all_preference_samples()
        assert len(samples) >= len(INDIC_PREFERENCES) + len(STEM_AND_GLOBAL_PREFERENCES)
        for s in samples:
            assert "prompt" in s and len(s["prompt"]) > 0
            assert "chosen" in s and len(s["chosen"]) > 0
            assert "rejected" in s and len(s["rejected"]) > 0

    def test_indic_languages_covered(self):
        indic_langs = {s.get("lang") for s in INDIC_PREFERENCES if "lang" in s}
        assert "hi" in indic_langs
        assert "bn" in indic_langs
        assert "ta" in indic_langs
        assert "te" in indic_langs
        assert "mr" in indic_langs
        assert "gu" in indic_langs
        assert "kn" in indic_langs
        assert "ml" in indic_langs

    def test_export_preference_curriculum(self, tmp_path: Path):
        out_file = tmp_path / "prefs.jsonl"
        count = export_preference_curriculum(out_file)
        assert count > 0
        assert out_file.is_file()

        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == count
        first = json.loads(lines[0])
        assert "prompt" in first
        assert "chosen" in first
        assert "rejected" in first

    def test_cli_parse_args(self):
        args = parse_args(["--output", "custom.jsonl", "--stats"])
        assert args.output == "custom.jsonl"
        assert args.stats is True

    def test_cli_main(self, tmp_path: Path):
        out_file = tmp_path / "curriculum_dpo.jsonl"
        code = pref_main(["--output", str(out_file), "--stats"])
        assert code == 0
        assert out_file.is_file()
