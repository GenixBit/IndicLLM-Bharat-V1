from __future__ import annotations

import json
from pathlib import Path

from bharat.data.world_knowledge import (
    CS_AND_AI_DATA,
    GLOBAL_SCIENCE_DATA,
    INDIC_22_LANGUAGES_DATA,
    WORLD_GEOGRAPHY_DATA,
    WORLD_HISTORY_DATA,
    export_world_knowledge_corpus,
    get_all_world_knowledge_documents,
    pack_world_knowledge_shards,
)
from bharat.tokenizer import load_tokenizer
from scripts.generate_world_curriculum import main as world_data_main
from scripts.generate_world_curriculum import parse_args


class TestWorldKnowledge:
    def test_curriculum_coverage(self):
        docs = get_all_world_knowledge_documents()
        assert len(docs) >= (
            len(GLOBAL_SCIENCE_DATA)
            + len(WORLD_HISTORY_DATA)
            + len(WORLD_GEOGRAPHY_DATA)
            + len(CS_AND_AI_DATA)
            + len(INDIC_22_LANGUAGES_DATA)
        )
        for d in docs:
            assert "category" in d
            assert "title" in d
            assert "text" in d and len(d["text"]) > 50

    def test_export_world_corpus(self, tmp_path: Path):
        out_p = tmp_path / "world_corpus.jsonl"
        count = export_world_knowledge_corpus(out_p)
        assert count > 0
        assert out_p.is_file()

        lines = out_p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == count
        first = json.loads(lines[0])
        assert "category" in first
        assert "title" in first

    def test_pack_world_shards(self, tmp_path: Path):
        tok = load_tokenizer("gpt2")
        out_dir = tmp_path / "shards"

        shards = pack_world_knowledge_shards(
            tokenizer=tok,
            output_dir=out_dir,
            max_tokens_per_shard=1000,
        )
        assert len(shards) > 0
        assert shards[0].is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--output", "custom_world.jsonl", "--pack"])
        assert args.output == "custom_world.jsonl"
        assert args.pack is True

    def test_cli_main(self, tmp_path: Path):
        out_p = tmp_path / "cli_world.jsonl"
        shards_dir = tmp_path / "cli_shards"

        code = world_data_main(
            [
                "--output",
                str(out_p),
                "--pack",
                "--shards-dir",
                str(shards_dir),
            ]
        )
        assert code == 0
        assert out_p.is_file()
        assert len(list(shards_dir.glob("*.bin"))) > 0
