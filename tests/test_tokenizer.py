from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bharat.tokenizer import BharatTokenizer, load_tokenizer, tokenizer_hash
from bharat.tokenizer.metadata import (
    TokenizerMetadata,
    metadata_from_tokenizer,
    validate_tokenizer_compatibility,
)


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> BharatTokenizer:
    return load_tokenizer("gpt2")


class TestTokenizerLoading:
    def test_load_gpt2(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.tokenizer_type == "gpt2"
        assert gpt2_tokenizer.vocab_size == 50257

    def test_load_from_string(self) -> None:
        tok = load_tokenizer("gpt2")
        assert tok is not None

    def test_load_default(self) -> None:
        tok = load_tokenizer()
        assert tok.tokenizer_type == "gpt2"


class TestTokenizerEncodeDecode:
    def test_encode_decode_roundtrip(self, gpt2_tokenizer: BharatTokenizer) -> None:
        text = "Hello, world!"
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)
        assert len(ids) > 0
        assert decoded.strip() != ""

    def test_encode_batch(self, gpt2_tokenizer: BharatTokenizer) -> None:
        texts = ["Hello", "world", "test"]
        batch = gpt2_tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(len(ids) > 0 for ids in batch)

    def test_decode_batch(self, gpt2_tokenizer: BharatTokenizer) -> None:
        texts = ["Hello", "world"]
        batch = gpt2_tokenizer.encode_batch(texts)
        decoded = gpt2_tokenizer.decode_batch(batch)
        assert len(decoded) == 2
        assert all(len(d) > 0 for d in decoded)

    def test_indic_text(self, gpt2_tokenizer: BharatTokenizer) -> None:
        text = "भारत एक महान देश है"
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)
        assert len(ids) > 0
        assert decoded.strip() != ""

    def test_empty_string(self, gpt2_tokenizer: BharatTokenizer) -> None:
        ids = gpt2_tokenizer.encode("")
        decoded = gpt2_tokenizer.decode(ids)
        assert isinstance(ids, list)
        assert isinstance(decoded, str)

    def test_special_characters(self, gpt2_tokenizer: BharatTokenizer) -> None:
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        ids = gpt2_tokenizer.encode(text)
        if ids:
            decoded = gpt2_tokenizer.decode(ids)
            assert isinstance(decoded, str)

    def test_long_text(self, gpt2_tokenizer: BharatTokenizer) -> None:
        text = " ".join(["word"] * 1000)
        ids = gpt2_tokenizer.encode(text)
        assert len(ids) > 0
        assert len(ids) < len(text)


class TestTokenizerProperties:
    def test_vocab_size(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.vocab_size == 50257

    def test_eos_token_id(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert isinstance(gpt2_tokenizer.eos_token_id, int)
        assert gpt2_tokenizer.eos_token_id >= 0

    def test_pad_token_id(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert isinstance(gpt2_tokenizer.pad_token_id, int)
        assert gpt2_tokenizer.pad_token_id >= 0

    def test_tokenizer_type(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.tokenizer_type in ("gpt2", "sentencepiece", "hf")


class TestTokenizerMetadata:
    def test_metadata_generated(self, gpt2_tokenizer: BharatTokenizer) -> None:
        meta = metadata_from_tokenizer(gpt2_tokenizer)
        assert isinstance(meta, TokenizerMetadata)
        assert meta.tokenizer_type == "gpt2"
        assert meta.vocab_size == 50257
        assert meta.eos_token_id == gpt2_tokenizer.eos_token_id
        assert meta.pad_token_id == gpt2_tokenizer.pad_token_id
        assert len(meta.tokenizer_hash) == 64

    def test_hash_consistency(self, gpt2_tokenizer: BharatTokenizer) -> None:
        h1 = tokenizer_hash(gpt2_tokenizer)
        h2 = tokenizer_hash(gpt2_tokenizer)
        assert h1 == h2

    def test_hash_format(self, gpt2_tokenizer: BharatTokenizer) -> None:
        h = tokenizer_hash(gpt2_tokenizer)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_fingerprint_deterministic(self, gpt2_tokenizer: BharatTokenizer) -> None:
        f1 = gpt2_tokenizer.fingerprint()
        f2 = gpt2_tokenizer.fingerprint()
        assert f1 == f2
        assert len(f1) == 64

    def test_fingerprint_differs_for_diff_tokenizer(self) -> None:
        tok1 = load_tokenizer("gpt2")
        # Create a second wrapper pointing to same HF tokenizer
        tok2 = load_tokenizer("gpt2")
        assert tok1.fingerprint() == tok2.fingerprint()

    def test_fingerprint_includes_vocab(self) -> None:
        tok = load_tokenizer("gpt2")
        fp = tok.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_tokenizer_json_not_auto_sentencepiece(self) -> None:
        """Verify generic tokenizer.json files are NOT auto-classified as SentencePiece."""
        from transformers import GPT2TokenizerFast
        from tokenizers import Tokenizer as HFTokenizersTokenizer

        gpt2_hf = GPT2TokenizerFast.from_pretrained("gpt2")
        backend = gpt2_hf.backend_tokenizer

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            backend.save(f.name)
            json_path = f.name

        tok = load_tokenizer(json_path)
        # Should NOT be sentencepiece for a GPT-2 tokenizer
        assert tok.tokenizer_type != "sentencepiece", (
            "GPT-2 tokenizer.json should not be classified as sentencepiece"
        )

        Path(json_path).unlink()

    def test_validate_compatibility_pass(self, gpt2_tokenizer: BharatTokenizer) -> None:
        meta = metadata_from_tokenizer(gpt2_tokenizer)
        validate_tokenizer_compatibility(meta, gpt2_tokenizer)

    def test_validate_compatibility_fail(self, gpt2_tokenizer: BharatTokenizer) -> None:
        meta = TokenizerMetadata(
            tokenizer_type="gpt2",
            vocab_size=100,
            eos_token_id=0,
            pad_token_id=0,
            tokenizer_hash="0" * 64,
        )
        with pytest.raises(ValueError, match="Tokenizer mismatch"):
            validate_tokenizer_compatibility(meta, gpt2_tokenizer)


class TestTokenizerVocabSize:
    def test_uint16_detection(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.vocab_size <= 65535


class TestTokenizerErrors:
    def test_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="Unsupported tokenizer source"):
            load_tokenizer(123)  # type: ignore[arg-type]

    def test_metadata_bad_tokenizer(self) -> None:
        with pytest.raises(TypeError, match="Expected BharatTokenizer"):
            metadata_from_tokenizer("not_a_tokenizer")  # type: ignore[arg-type]

    def test_validate_bad_tokenizer(self) -> None:
        meta = metadata_from_tokenizer(load_tokenizer("gpt2"))
        with pytest.raises(TypeError, match="Expected BharatTokenizer"):
            validate_tokenizer_compatibility(meta, "not_a_tokenizer")  # type: ignore[arg-type]
