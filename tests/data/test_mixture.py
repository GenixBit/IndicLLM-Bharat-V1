from __future__ import annotations

from pathlib import Path

from bharat.data.mixture import (
    MixtureWeights,
    build_world_and_indic_corpus,
    clean_and_filter_text,
    stream_and_pack_mixture,
)
from bharat.tokenizer import load_tokenizer
from scripts.prepare_mixture_shards import main as prepare_shards_main
from scripts.prepare_mixture_shards import parse_args


class TestDataMixture:
    def test_mixture_weights_validation(self):
        w = MixtureWeights()
        w.validate()  # Should not raise

        bad_w = MixtureWeights(indic_multilingual=0.8)
        import pytest

        with pytest.raises(ValueError, match="Mixture weights must sum to 1.0"):
            bad_w.validate()

    def test_clean_and_filter_text(self):
        sample = "  This is a   clean scientific sentence about quantum mechanics. Contact me at test@example.com!  "
        cleaned = clean_and_filter_text(sample, min_chars=20)
        assert "[EMAIL]" in cleaned
        assert "  " not in cleaned

        short = "Too short"
        assert clean_and_filter_text(short, min_chars=50) == ""

    def test_build_corpus(self):
        corpus = build_world_and_indic_corpus()
        assert len(corpus) > 10
        assert any("def binary_search" in c for c in corpus)

    def test_stream_and_pack_mixture(self, tmp_path: Path):
        tok = load_tokenizer("gpt2")
        shards = stream_and_pack_mixture(
            tokenizer=tok,
            output_dir=tmp_path,
            max_tokens_per_shard=1000,
            max_docs=5,
        )
        assert len(shards) > 0
        assert shards[0].is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--output-dir", "custom_shards", "--max-docs", "10"])
        assert args.output_dir == "custom_shards"
        assert args.max_docs == 10

    def test_cli_main(self, tmp_path: Path):
        code = prepare_shards_main(["--output-dir", str(tmp_path), "--max-docs", "3"])
        assert code == 0
        assert len(list(tmp_path.glob("*.bin"))) > 0
