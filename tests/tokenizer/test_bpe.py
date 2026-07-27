from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import (
    BPETokenizer,
    _BPEMerge,
    _build_byte_vocab,
    compute_tokenizer_hash,
    train_bpe,
)

_SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"text": "hello world"}) + "\n",
        json.dumps({"text": "bpe tokenizer"}) + "\n",
        json.dumps({"text": "deterministic"}) + "\n",
        json.dumps({"text": "byte level bpe"}) + "\n",
        json.dumps({"text": "hello again world"}) + "\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ── byte vocabulary ────────────────────────────────────────────────


def test_byte_vocab_includes_special_tokens() -> None:
    vocab = _build_byte_vocab(_SPECIAL)
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<bos>"] == 2
    assert vocab["<eos>"] == 3
    assert len(vocab) == 260


def test_byte_vocab_all_256_bytes() -> None:
    vocab = _build_byte_vocab(_SPECIAL)
    for b in range(256):
        token_str = f"<byte_{b:02x}>"
        assert token_str in vocab
    assert len(vocab) == 4 + 256


def test_byte_vocab_ids_contiguous_after_special() -> None:
    vocab = _build_byte_vocab(_SPECIAL)
    expected_ids = list(range(260))
    actual_ids = sorted(vocab.values())
    assert actual_ids == expected_ids


def test_byte_vocab_empty_special() -> None:
    vocab = _build_byte_vocab({})
    assert len(vocab) == 256, "should have exactly 256 byte tokens"
    for b in range(256):
        assert f"<byte_{b:02x}>" in vocab


# ── training ────────────────────────────────────────────────────────


def test_train_bpe_returns_bpe_tokenizer(tiny_corpus: Path) -> None:
    result = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    assert isinstance(result, BPETokenizer)
    assert len(result.vocab) == 270
    assert len(result.merges) == 10
    assert result.special_tokens == _SPECIAL


def test_train_bpe_minimal_vocab(tiny_corpus: Path) -> None:
    result = train_bpe(tiny_corpus, vocab_size=260, special_tokens=_SPECIAL)
    assert len(result.vocab) == 260
    assert len(result.merges) == 0


def test_train_bpe_empty_corpus(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    result = train_bpe(path, vocab_size=260, special_tokens=_SPECIAL)
    assert len(result.vocab) == 260


def test_train_bpe_save_and_load(tiny_corpus: Path, tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    t1 = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    t1.save(tokenizer_path)
    t2 = BPETokenizer.load(tokenizer_path)
    assert t1.vocab == t2.vocab
    assert t1.merges == t2.merges
    assert t1.special_tokens == t2.special_tokens
    assert t1.tokenizer_hash == t2.tokenizer_hash


# ── determinism ────────────────────────────────────────────────────


def test_train_bpe_deterministic_vocab(tiny_corpus: Path) -> None:
    r1 = train_bpe(tiny_corpus, vocab_size=280, special_tokens=_SPECIAL)
    for _ in range(3):
        r2 = train_bpe(tiny_corpus, vocab_size=280, special_tokens=_SPECIAL)
        assert r1.vocab == r2.vocab, "vocab differs between runs"
        assert r1.merges == r2.merges, "merges differ between runs"
        assert r1.tokenizer_hash == r2.tokenizer_hash, "hash differs between runs"


def test_train_bpe_deterministic_with_delay(tiny_corpus: Path) -> None:
    r1 = train_bpe(tiny_corpus, vocab_size=280, special_tokens=_SPECIAL)
    time.sleep(3)
    r2 = train_bpe(tiny_corpus, vocab_size=280, special_tokens=_SPECIAL)
    assert r1.vocab == r2.vocab
    assert r1.merges == r2.merges
    assert r1.tokenizer_hash == r2.tokenizer_hash


def test_train_bpe_different_vocab_size_changes_hash(tiny_corpus: Path) -> None:
    r1 = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    r2 = train_bpe(tiny_corpus, vocab_size=280, special_tokens=_SPECIAL)
    assert r1.tokenizer_hash != r2.tokenizer_hash


def test_train_bpe_different_special_tokens_changes_hash(tiny_corpus: Path) -> None:
    r1 = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    alt = {"<pad>": 0, "<unk>": 1}
    r2 = train_bpe(tiny_corpus, vocab_size=270, special_tokens=alt)
    assert r1.tokenizer_hash != r2.tokenizer_hash


def test_train_bpe_different_corpus_changes_hash(tmp_path: Path) -> None:
    corpus_a = tmp_path / "a.jsonl"
    corpus_a.write_text(json.dumps({"text": "alpha beta gamma"}) + "\n", encoding="utf-8")
    corpus_b = tmp_path / "b.jsonl"
    corpus_b.write_text(json.dumps({"text": "delta epsilon zeta"}) + "\n", encoding="utf-8")
    r1 = train_bpe(corpus_a, vocab_size=270, special_tokens=_SPECIAL)
    r2 = train_bpe(corpus_b, vocab_size=270, special_tokens=_SPECIAL)
    assert r1.tokenizer_hash != r2.tokenizer_hash


# ── serialization ──────────────────────────────────────────────────


def test_to_dict_round_trip(tiny_corpus: Path) -> None:
    t = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    data = t.to_dict()
    assert "vocab" in data
    assert "merges" in data
    assert "special_tokens" in data
    assert "tokenizer_hash" in data
    reconstructed = BPETokenizer.from_dict(data)
    assert t.vocab == reconstructed.vocab
    assert t.merges == reconstructed.merges
    assert t.tokenizer_hash == reconstructed.tokenizer_hash


def test_to_dict_merges_are_tuples_of_three(tiny_corpus: Path) -> None:
    t = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    data = t.to_dict()
    for m in data["merges"]:
        assert len(m) == 3
        assert isinstance(m[0], int)
        assert isinstance(m[1], int)
        assert isinstance(m[2], int)


def test_save_produces_valid_json(tiny_corpus: Path, tmp_path: Path) -> None:
    t = train_bpe(tiny_corpus, vocab_size=270, special_tokens=_SPECIAL)
    path = tmp_path / "t.json"
    t.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "vocab" in data
    assert "merges" in data


# ── tokenizer_hash ─────────────────────────────────────────────────


def test_compute_tokenizer_hash_deterministic() -> None:
    merges = (_BPEMerge(left=97, right=98, token=260),)
    vocab = {"<byte_61>": 260, "<byte_62>": 97, "<byte_63>": 98}
    h1 = compute_tokenizer_hash(vocab, merges)
    for _ in range(3):
        assert compute_tokenizer_hash(vocab, merges) == h1


def test_hash_changes_when_merges_change() -> None:
    base_merges = (_BPEMerge(left=97, right=98, token=260),)
    other_merges = (_BPEMerge(left=99, right=100, token=260),)
    vocab = {"<byte_61>": 260, "<byte_62>": 97, "<byte_63>": 98}
    h1 = compute_tokenizer_hash(vocab, base_merges)
    h2 = compute_tokenizer_hash(vocab, other_merges)
    assert h1 != h2


def test_hash_changes_when_vocab_changes() -> None:
    merges = (_BPEMerge(left=97, right=98, token=260),)
    vocab_a = {"<byte_61>": 260, "<byte_62>": 97}
    vocab_b = {"<byte_61>": 261, "<byte_62>": 97}
    assert compute_tokenizer_hash(vocab_a, merges) != compute_tokenizer_hash(vocab_b, merges)


def test_hash_empty_merges() -> None:
    vocab = _build_byte_vocab(_SPECIAL)
    h = compute_tokenizer_hash(vocab, ())
    assert isinstance(h, str)
    assert len(h) == 64
