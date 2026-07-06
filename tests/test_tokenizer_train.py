from __future__ import annotations

from typing import ClassVar

import pytest

from bharat.tokenizer.train import train_bpe_tokenizer


class TestTrainBPETokenizer:
    SAMPLE_TEXTS: ClassVar[list[str]] = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",
        "Machine learning is transforming artificial intelligence.",
        "भारत एक महान देश है",
        "বাংলা ভাষা একটি সুন্দর ভাষা",
    ]

    def test_train_small_bpe(self, tmp_path: pytest.TempPathFactory) -> None:
        tok = train_bpe_tokenizer(
            self.SAMPLE_TEXTS,
            vocab_size=500,
            output_dir=str(tmp_path / "bpe_tok"),
        )
        assert tok.tokenizer_type == "sentencepiece"
        assert tok.vocab_size <= 500

    def test_train_saves_to_disk(self, tmp_path: pytest.TempPathFactory) -> None:
        output = tmp_path / "bpe_tok"
        tok = train_bpe_tokenizer(
            self.SAMPLE_TEXTS,
            vocab_size=500,
            output_dir=str(output),
        )
        assert tok is not None
        assert (output / "tokenizer.json").exists()

    def test_encode_after_train(self, tmp_path: pytest.TempPathFactory) -> None:
        tok = train_bpe_tokenizer(
            self.SAMPLE_TEXTS,
            vocab_size=500,
            output_dir=str(tmp_path / "bpe_tok"),
        )
        ids = tok.encode("Hello world")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert isinstance(decoded, str) and len(decoded) > 0

    def test_train_without_output(self) -> None:
        tok = train_bpe_tokenizer(
            self.SAMPLE_TEXTS,
            vocab_size=500,
        )
        ids = tok.encode("Test")
        assert len(ids) > 0
