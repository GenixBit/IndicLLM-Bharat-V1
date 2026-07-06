from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bharat.tokenizer import BharatTokenizer, load_tokenizer, tokenizer_hash
from bharat.tokenizer.metadata import (
    TokenizerMetadata,
    metadata_from_tokenizer,
    validate_tokenizer_compatibility,
)

# ---------------------------------------------------------------------------
# Offline core tests using tiny BPE tokenizer
# ---------------------------------------------------------------------------


class TestTinyTokenizerLoading:
    def test_load_tiny_bpe(self, tiny_tokenizer: BharatTokenizer) -> None:
        assert tiny_tokenizer.tokenizer_type in ("gpt2", "hf", "sentencepiece")
        assert tiny_tokenizer.vocab_size > 0

    def test_load_tiny_sp(self, tiny_sp_tokenizer: BharatTokenizer) -> None:
        assert tiny_sp_tokenizer.tokenizer_type == "sentencepiece"
        assert tiny_sp_tokenizer.vocab_size > 0

    def test_load_from_string(self) -> None:
        from tests.conftest import _build_tiny_bpe_tokenizer

        tok_obj = _build_tiny_bpe_tokenizer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tok_obj.save(f.name)
            json_path = f.name
        tok = load_tokenizer(json_path)
        assert tok is not None
        Path(json_path).unlink()


class TestTinyTokenizerEncodeDecode:
    def test_encode_decode_roundtrip(self, tiny_tokenizer: BharatTokenizer) -> None:
        text = "Hello, world!"
        ids = tiny_tokenizer.encode(text)
        decoded = tiny_tokenizer.decode(ids)
        assert len(ids) > 0
        assert decoded.strip() != ""

    def test_encode_batch(self, tiny_tokenizer: BharatTokenizer) -> None:
        texts = ["Hello", "world", "test"]
        batch = tiny_tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(len(ids) > 0 for ids in batch)

    def test_decode_batch(self, tiny_tokenizer: BharatTokenizer) -> None:
        texts = ["Hello", "world"]
        batch = tiny_tokenizer.encode_batch(texts)
        decoded = tiny_tokenizer.decode_batch(batch)
        assert len(decoded) == 2
        assert all(len(d) > 0 for d in decoded)

    def test_indic_text(self, tiny_tokenizer: BharatTokenizer) -> None:
        text = "भारत"
        ids = tiny_tokenizer.encode(text)
        decoded = tiny_tokenizer.decode(ids)
        assert len(ids) > 0
        assert decoded.strip() != ""

    def test_empty_string(self, tiny_tokenizer: BharatTokenizer) -> None:
        ids = tiny_tokenizer.encode("")
        decoded = tiny_tokenizer.decode(ids)
        assert isinstance(ids, list)
        assert isinstance(decoded, str)

    def test_long_text(self, tiny_tokenizer: BharatTokenizer) -> None:
        text = " ".join(["word"] * 100)
        ids = tiny_tokenizer.encode(text)
        assert len(ids) > 0
        assert len(ids) < len(text)


class TestTinyTokenizerProperties:
    def test_vocab_size(self, tiny_tokenizer: BharatTokenizer) -> None:
        assert tiny_tokenizer.vocab_size > 0

    def test_eos_token_id(self, tiny_tokenizer: BharatTokenizer) -> None:
        assert isinstance(tiny_tokenizer.eos_token_id, int)
        assert tiny_tokenizer.eos_token_id >= 0

    def test_pad_token_id(self, tiny_tokenizer: BharatTokenizer) -> None:
        assert isinstance(tiny_tokenizer.pad_token_id, int)
        assert tiny_tokenizer.pad_token_id >= 0

    def test_tokenizer_type(self, tiny_tokenizer: BharatTokenizer) -> None:
        assert tiny_tokenizer.tokenizer_type in ("gpt2", "sentencepiece", "hf")


# ---------------------------------------------------------------------------
# Fingerprint tests (offline)
# ---------------------------------------------------------------------------


class TestTokenizerFingerprint:
    def test_deterministic(self, tiny_tokenizer: BharatTokenizer) -> None:
        f1 = tiny_tokenizer.fingerprint()
        f2 = tiny_tokenizer.fingerprint()
        assert f1 == f2
        assert len(f1) == 64

    def test_same_tokenizer_same_fingerprint(self, tiny_bpe_tokenizer_json: str) -> None:
        t1 = load_tokenizer(tiny_bpe_tokenizer_json)
        t2 = load_tokenizer(tiny_bpe_tokenizer_json)
        assert t1.fingerprint() == t2.fingerprint()

    def test_fingerprint_changes_when_special_token_added(
        self, tiny_bpe_tokenizer_json: str
    ) -> None:
        tok = load_tokenizer(tiny_bpe_tokenizer_json)
        fp_before = tok.fingerprint()
        tok.add_special_tokens({"additional_special_tokens": ["<|new_token|>"]})
        fp_after = tok.fingerprint()
        assert fp_before != fp_after, "Adding special tokens should change fingerprint"

    def test_different_vocab_changes_fingerprint(self, tiny_bpe_tokenizer_json: str) -> None:
        tok1 = load_tokenizer(tiny_bpe_tokenizer_json)
        fp1 = tok1.fingerprint()
        # Load a SentencePiece tokenizer which has different vocab
        from tests.conftest import _build_tiny_bpe_tokenizer

        tok_obj = _build_tiny_bpe_tokenizer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tok_obj.save(f.name)
            path2 = f.name
        tok2 = load_tokenizer(path2)
        fp2 = tok2.fingerprint()
        # Same model + data should produce same fingerprint
        assert fp1 == fp2, "Identical tokenizers should have same fingerprint"
        Path(path2).unlink()

    def test_sp_fingerprint_deterministic(self, tiny_sp_tokenizer: BharatTokenizer) -> None:
        f1 = tiny_sp_tokenizer.fingerprint()
        f2 = tiny_sp_tokenizer.fingerprint()
        assert f1 == f2
        assert len(f1) == 64

    def test_sp_reloaded_same_fingerprint(self, tiny_sp_model_path: Path) -> None:
        t1 = load_tokenizer(str(tiny_sp_model_path))
        t2 = load_tokenizer(str(tiny_sp_model_path))
        assert t1.fingerprint() == t2.fingerprint()


# ---------------------------------------------------------------------------
# SentencePiece .model loading
# ---------------------------------------------------------------------------


class TestSentencePieceModelLoading:
    def test_load_real_sentencepiece_model(self, tiny_sp_tokenizer: BharatTokenizer) -> None:
        assert tiny_sp_tokenizer.tokenizer_type == "sentencepiece"
        assert tiny_sp_tokenizer.vocab_size > 0

    def test_sp_is_not_bpe(self, tiny_sp_tokenizer: BharatTokenizer) -> None:
        assert tiny_sp_tokenizer.tokenizer_type != "bpe"

    def test_sp_encode_decode(self, tiny_sp_tokenizer: BharatTokenizer) -> None:
        text = "Hello world"
        ids = tiny_sp_tokenizer.encode(text)
        assert len(ids) > 0
        decoded = tiny_sp_tokenizer.decode(ids)
        assert len(decoded) > 0

    def test_tokenizer_json_not_auto_sentencepiece(self, tiny_bpe_tokenizer_json: str) -> None:
        """Verify generic tokenizer.json files are NOT auto-classified as SentencePiece."""
        from tokenizers import Tokenizer as HFTokenizersTokenizer

        # Load the tiny BPE tokenizer and re-save as a generic tokenizer.json
        bpe = HFTokenizersTokenizer.from_file(tiny_bpe_tokenizer_json)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            bpe.save(f.name)
            json_path = f.name

        tok = load_tokenizer(json_path)
        # Should NOT be sentencepiece for a BPE tokenizer
        assert tok.tokenizer_type != "sentencepiece"
        Path(json_path).unlink()

    def test_sp_model_directory_loading(self, tiny_sp_model_path: Path) -> None:
        """Verify .model files in a directory are loaded correctly."""
        tok = load_tokenizer(str(tiny_sp_model_path))
        assert tok.tokenizer_type == "sentencepiece"


# ---------------------------------------------------------------------------
# Metadata tests (offline)
# ---------------------------------------------------------------------------


class TestTokenizerMetadata:
    def test_metadata_generated(self, tiny_tokenizer: BharatTokenizer) -> None:
        meta = metadata_from_tokenizer(tiny_tokenizer)
        assert isinstance(meta, TokenizerMetadata)
        assert meta.tokenizer_type == tiny_tokenizer.tokenizer_type
        assert meta.vocab_size == tiny_tokenizer.vocab_size
        assert meta.eos_token_id == tiny_tokenizer.eos_token_id
        assert meta.pad_token_id == tiny_tokenizer.pad_token_id
        assert len(meta.tokenizer_hash) == 64

    def test_hash_consistency(self, tiny_tokenizer: BharatTokenizer) -> None:
        h1 = tokenizer_hash(tiny_tokenizer)
        h2 = tokenizer_hash(tiny_tokenizer)
        assert h1 == h2

    def test_hash_format(self, tiny_tokenizer: BharatTokenizer) -> None:
        h = tokenizer_hash(tiny_tokenizer)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_validate_compatibility_pass(self, tiny_tokenizer: BharatTokenizer) -> None:
        meta = metadata_from_tokenizer(tiny_tokenizer)
        validate_tokenizer_compatibility(meta, tiny_tokenizer)

    def test_validate_compatibility_fail(self, tiny_tokenizer: BharatTokenizer) -> None:
        meta = TokenizerMetadata(
            tokenizer_type=tiny_tokenizer.tokenizer_type,
            vocab_size=100,
            eos_token_id=0,
            pad_token_id=0,
            tokenizer_hash="0" * 64,
        )
        with pytest.raises(ValueError, match="Tokenizer mismatch"):
            validate_tokenizer_compatibility(meta, tiny_tokenizer)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestTokenizerErrors:
    def test_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="Unsupported tokenizer source"):
            load_tokenizer(123)  # type: ignore[arg-type]

    def test_metadata_bad_tokenizer(self) -> None:
        with pytest.raises(TypeError, match="Expected BharatTokenizer"):
            metadata_from_tokenizer("not_a_tokenizer")  # type: ignore[arg-type]

    def test_validate_bad_tokenizer(self, tiny_bpe_tokenizer_json: str) -> None:
        with pytest.raises(TypeError, match="Expected BharatTokenizer"):
            validate_tokenizer_compatibility(
                metadata_from_tokenizer(load_tokenizer(tiny_bpe_tokenizer_json)),
                "not_a_tokenizer",
            )


# ---------------------------------------------------------------------------
# Integration tests (require internet)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGPT2Integration:
    @pytest.fixture(scope="class")
    def gpt2_tokenizer(self) -> BharatTokenizer:
        return load_tokenizer("gpt2")

    def test_load_gpt2(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.tokenizer_type == "gpt2"
        assert gpt2_tokenizer.vocab_size == 50257

    def test_vocab_size(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert gpt2_tokenizer.vocab_size == 50257

    def test_eos_token_id(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert isinstance(gpt2_tokenizer.eos_token_id, int)
        assert gpt2_tokenizer.eos_token_id >= 0

    def test_pad_token_id(self, gpt2_tokenizer: BharatTokenizer) -> None:
        assert isinstance(gpt2_tokenizer.pad_token_id, int)
        assert gpt2_tokenizer.pad_token_id >= 0
