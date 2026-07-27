from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.tokenizer import BharatBPETokenizer, BharatTokenizer, train_bpe


def _build_tokenizer(tmp_path: Path) -> BharatBPETokenizer:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": text}) for text in ["भारत", "hello", "भारत hello"]) + "\n",
        encoding="utf-8",
    )
    return BharatBPETokenizer(train_bpe(corpus, vocab_size=272))


def test_adapter_implements_bharat_tokenizer(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer(tmp_path)

    assert isinstance(tokenizer, BharatTokenizer)
    assert tokenizer.tokenizer_type == "bharat_bpe"
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3


def test_encode_adds_bos_and_eos_only_when_requested(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer(tmp_path)

    plain = tokenizer.encode("भारत", add_special_tokens=False)
    wrapped = tokenizer.encode("भारत", add_special_tokens=True)

    assert wrapped == [tokenizer.bos_token_id, *plain, tokenizer.eos_token_id]
    assert tokenizer.decode(wrapped) == "भारत"


def test_batch_methods_preserve_order(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer(tmp_path)
    texts = ["hello", "भारत"]

    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)

    assert encoded == [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    assert tokenizer.decode_batch(encoded) == texts


def test_metadata_and_fingerprint_are_deterministic(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer(tmp_path)

    metadata = tokenizer.get_metadata()

    assert metadata["fingerprint"] == tokenizer.fingerprint()
    assert metadata["vocab_size"] == tokenizer.vocab_size
    assert metadata["normalization"] == "nfc"
    assert tokenizer.fingerprint() == tokenizer.fingerprint()


def test_load_round_trip(tmp_path: Path) -> None:
    tokenizer = _build_tokenizer(tmp_path)
    artifact = tmp_path / "tokenizer.json"
    tokenizer.save(artifact)

    loaded = BharatBPETokenizer.load(artifact)

    assert loaded.fingerprint() == tokenizer.fingerprint()
    assert loaded.encode("hello") == tokenizer.encode("hello")


def test_missing_required_special_token_fails_loudly(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello"}\n', encoding="utf-8")
    raw = train_bpe(
        corpus,
        vocab_size=260,
        special_tokens={"<pad>": 0, "<unk>": 1, "<bos>": 2},
    )
    tokenizer = BharatBPETokenizer(raw)

    with pytest.raises(ValueError, match="missing required special token '<eos>'"):
        _ = tokenizer.eos_token_id
