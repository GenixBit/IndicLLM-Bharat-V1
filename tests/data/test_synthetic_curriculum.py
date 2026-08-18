from __future__ import annotations

import json
from pathlib import Path

from bharat.data.synthetic_curriculum import (
    INDIC_LANGUAGES,
    export_curriculum_datasets,
    generate_curriculum,
)


class TestSyntheticCurriculum:
    def test_indic_languages_count(self):
        assert len(INDIC_LANGUAGES) == 22

    def test_generate_curriculum(self):
        samples = generate_curriculum(num_samples=50)
        assert len(samples) == 50
        for s in samples:
            assert "id" in s
            assert "domain" in s
            assert "language" in s
            assert "text" in s
            assert "instruction" in s
            assert "response" in s
            assert len(s["text"]) > 10

    def test_export_curriculum_datasets(self, tmp_path: Path):
        pretrain_p, sft_p = export_curriculum_datasets(tmp_path, num_samples=25)
        assert pretrain_p.is_file()
        assert sft_p.is_file()

        with open(sft_p, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 25
        assert "messages" in lines[0]
        assert len(lines[0]["messages"]) == 3
