from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import (
    _SPECIAL_TOKENS,
    BPETokenizer,
    _build_base_vocab,
    _read_corpus_records,
    _validate_special_tokens,
    _validate_vocab_size,
    train_bpe,
)

_SPECIAL = dict(_SPECIAL_TOKENS)


# ── helpers ────────────────────────────────────────────────────────


def _write_jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def corpus_en(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "corpus.jsonl",
        ["hello world", "bpe tokenizer", "deterministic", "byte level bpe", "hello again world"],
    )


# ══════════════════════════════════════════════════════════════════
# 1. Validation
# ══════════════════════════════════════════════════════════════════


def test_no_special_byte_id_collision() -> None:
    byte_vid, id2b, vocab = _build_base_vocab(_SPECIAL, {})
    all_byte_ids = set(byte_vid.values())
    all_special_ids = set(_SPECIAL.values())
    assert all_byte_ids.isdisjoint(all_special_ids)


def test_all_256_byte_values_have_valid_ids() -> None:
    byte_vid, id2b, vocab = _build_base_vocab(_SPECIAL, {})
    assert len(byte_vid) == 256
    for b in range(256):
        tid = byte_vid[b]
        assert tid in id2b
        assert id2b[tid] == bytes([b])


def test_byte_id_mapping_survives_save_load(tmp_path: Path) -> None:
    byte_vid, id2b, vocab = _build_base_vocab(_SPECIAL, {})
    t = BPETokenizer(
        byte_value_to_id=byte_vid,
        id_to_bytes=id2b,
        vocab=vocab,
    )
    t.tokenizer_hash = t.compute_hash()
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    t2 = BPETokenizer.load(p)
    assert t.byte_value_to_id == t2.byte_value_to_id
    assert t.id_to_bytes == t2.id_to_bytes


def test_duplicate_special_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate special token ID"):
        _validate_special_tokens({"<a>": 0, "<b>": 0})


def test_empty_special_string_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_special_tokens({"": 0})


def test_negative_special_id_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _validate_special_tokens({"<a>": -1})


def test_vocab_size_below_base_rejected() -> None:
    with pytest.raises(ValueError, match="less than base vocabulary size"):
        _validate_vocab_size(4, 260)


def test_vocab_size_at_base_accepted() -> None:
    _validate_vocab_size(260, 260)


def test_vocab_size_above_base_accepted() -> None:
    _validate_vocab_size(300, 260)


# ══════════════════════════════════════════════════════════════════
# 2. Training
# ══════════════════════════════════════════════════════════════════


def test_train_returns_bpe_tokenizer(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    assert isinstance(t, BPETokenizer)
    assert t.vocab_size == 280


def test_train_minimal_vocab(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    assert t.vocab_size == 260
    assert len(t.merges) == 0


def test_train_empty_corpus(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "empty.jsonl", [])
    t = train_bpe(path, vocab_size=260)
    assert t.vocab_size == 260


def test_unreachable_vocab_reports_actual_size(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=100000)
    actual = t.vocab_size
    assert actual < 100000
    assert actual > 260
    assert len(t.merges) > 0


def test_merges_are_used_by_encode(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    ids = t.encode("hello world")
    assert len(ids) < len(b"hello world")


def test_decode_reconstructs_merged_tokens(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    text = "hello world"
    ids = t.encode(text)
    decoded = t.decode(ids)
    assert decoded == text


# ══════════════════════════════════════════════════════════════════
# 3. Round-trip by language
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "नमस्ते दुनिया",
        "नमस्कार जग",
        "ওহে বিশ্ব",
        "હેલો વર્લ્ડ",
        "வணக்கம் உலகம்",
        "హలో వరల్డ్",
        "ನಮಸ್ಕಾರ ವರ್ಲ್ಡ್",
        "ഹലോ വേൾഡ്",
        "ନମସ୍କାର ବିଶ୍ୱ",
        "😀🌍🎉",
        "hello\tworld\nline2",
        "  spaces  ",
    ],
)
def test_round_trip(tmp_path: Path, text: str) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", [text, "padding for merges"])
    t = train_bpe(path, vocab_size=280)
    ids = t.encode(text)
    decoded = t.decode(ids)
    assert decoded == text


# ══════════════════════════════════════════════════════════════════
# 4. Determinism
# ══════════════════════════════════════════════════════════════════


def test_repeated_vocab_equality(corpus_en: Path) -> None:
    r1 = train_bpe(corpus_en, vocab_size=280)
    for _ in range(3):
        r2 = train_bpe(corpus_en, vocab_size=280)
        assert r1.vocab == r2.vocab


def test_repeated_merge_equality(corpus_en: Path) -> None:
    r1 = train_bpe(corpus_en, vocab_size=280)
    for _ in range(3):
        r2 = train_bpe(corpus_en, vocab_size=280)
        assert r1.merges == r2.merges


def test_repeated_hash_equality(corpus_en: Path) -> None:
    r1 = train_bpe(corpus_en, vocab_size=280)
    for _ in range(3):
        r2 = train_bpe(corpus_en, vocab_size=280)
        assert r1.tokenizer_hash == r2.tokenizer_hash


def test_deterministic_with_delay(corpus_en: Path) -> None:
    r1 = train_bpe(corpus_en, vocab_size=280)
    time.sleep(3)
    r2 = train_bpe(corpus_en, vocab_size=280)
    assert r1.vocab == r2.vocab
    assert r1.merges == r2.merges
    assert r1.tokenizer_hash == r2.tokenizer_hash


def test_byte_identical_artifact(corpus_en: Path, tmp_path: Path) -> None:
    p1 = tmp_path / "t1.json"
    p2 = tmp_path / "t2.json"
    train_bpe(corpus_en, vocab_size=280).save(p1, overwrite=True)
    train_bpe(corpus_en, vocab_size=280).save(p2, overwrite=True)
    assert p1.read_bytes() == p2.read_bytes()


# ══════════════════════════════════════════════════════════════════
# 5. Encode/decode
# ══════════════════════════════════════════════════════════════════


def test_unknown_token_in_decode_rejected(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    with pytest.raises(ValueError, match="unknown token ID"):
        t.decode([99999])


def test_inactive_special_not_emitted(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    ids = t.encode("hello world")
    for sid in _SPECIAL.values():
        assert sid not in ids


def test_special_parsing_only_when_enabled(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    text = "<pad>"
    ids_no_special = t.encode(text, allow_special=False)
    assert t.decode(ids_no_special) == "<pad>"
    ids_special = t.encode(text, allow_special=True)
    assert _SPECIAL["<pad>"] in ids_special


def test_encode_lone_surrogate_rejected(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    with pytest.raises(ValueError, match="lone surrogate"):
        t.encode("\ud800")


# ══════════════════════════════════════════════════════════════════
# 6. JSONL reading
# ══════════════════════════════════════════════════════════════════


def test_json_syntax_not_included_as_training_text(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", ["hello"])
    raw = path.read_bytes()
    records = _read_corpus_records(path)
    total_bytes = sum(len(r) for r in records)
    assert total_bytes < len(raw)


def test_pairs_do_not_cross_records(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", ["ab", "cd"])
    records = _read_corpus_records(path)
    all_ids = [b for rec in records for b in rec]
    pair_at_boundary = (all_ids[1], all_ids[2])
    t = train_bpe(path, vocab_size=270)
    for merge in t.merges:
        assert (merge.left, merge.right) != pair_at_boundary


def test_malformed_jsonl_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        train_bpe(path, vocab_size=260)


def test_malformed_utf8_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"text": "\xff"}\n')
    with pytest.raises(ValueError, match="malformed UTF-8"):
        train_bpe(path, vocab_size=260)


def test_lone_surrogate_in_jsonl_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"text": "\\ud800"}\n')
    with pytest.raises(ValueError, match="lone surrogate"):
        train_bpe(path, vocab_size=260)


def test_non_object_record_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('"just a string"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        train_bpe(path, vocab_size=260)


def test_missing_text_field_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"nope": "hello"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'text' field"):
        train_bpe(path, vocab_size=260)


def test_non_string_text_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": 42}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a string"):
        train_bpe(path, vocab_size=260)


# ══════════════════════════════════════════════════════════════════
# 7. Serialization
# ══════════════════════════════════════════════════════════════════


def test_save_load_id_equality(corpus_en: Path, tmp_path: Path) -> None:
    t1 = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t1.save(p, overwrite=True)
    t2 = BPETokenizer.load(p)
    assert t1.byte_value_to_id == t2.byte_value_to_id
    assert t1.id_to_bytes == t2.id_to_bytes
    assert t1.merges == t2.merges
    assert t1.vocab == t2.vocab
    assert t1.special_tokens == t2.special_tokens
    assert t1.tokenizer_hash == t2.tokenizer_hash


def test_hash_tampering_rejected(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["tokenizer_hash"] = "0" * 64
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        BPETokenizer.load(p)


def test_save_no_overwrite_by_default(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    with pytest.raises(FileExistsError):
        t.save(p)


def test_save_overwrite_ok(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    t.save(p, overwrite=True)


def test_unsupported_schema_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema_version": "bogus-v2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema version"):
        BPETokenizer.load(p)


# ══════════════════════════════════════════════════════════════════
# 8. CPU-only and offline execution
# ══════════════════════════════════════════════════════════════════


def test_cpu_only_offline(corpus_en: Path) -> None:
    train_bpe(corpus_en, vocab_size=280)
