from __future__ import annotations

import dataclasses
import json
import time
import unicodedata
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import (
    _SPECIAL_TOKENS,
    BPETokenizer,
    _build_base_vocab,
    _read_corpus_records,
    _validate_special_and_reserved_tokens,
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
    with pytest.raises(ValueError, match="duplicate token ID 0"):
        _validate_special_and_reserved_tokens({"<a>": 0, "<b>": 0}, {})


def test_empty_special_string_rejected() -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _validate_special_and_reserved_tokens({"": 0}, {})


def test_negative_special_id_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_special_and_reserved_tokens({"<a>": -1}, {})


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


# ══════════════════════════════════════════════════════════════════
# 9. NFC normalization
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "decomposed,composed",
    [
        ("e\u0301", "\u00e9"),
        ("a\u0301", "\u00e1"),
        ("\u0041\u0300", "\u00c0"),
        ("\u0041\u0301", "\u00c1"),
        ("\u0041\u0302", "\u00c2"),
        ("\u006f\u0308", "\u00f6"),
        ("\u004e\u0303", "\u00d1"),
        ("\u0063\u0327", "\u00e7"),
    ],
)
def test_nfc_normalization(tmp_path: Path, decomposed: str, composed: str) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(f'{{"text": "{decomposed}"}}\n', encoding="utf-8")
    t = train_bpe(path, vocab_size=300)
    encoded = t.encode(composed)
    decoded = t.decode(encoded)
    assert decoded == composed


def test_nfc_normalization_decode_encode_non_nfc(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    decomposed = "e\u0301"
    path.write_text(f'{{"text": "{decomposed}"}}\n', encoding="utf-8")
    t = train_bpe(path, vocab_size=300)
    encoded = t.encode(decomposed)
    decoded = t.decode(encoded)
    assert decoded == unicodedata.normalize("NFC", decomposed)


# ══════════════════════════════════════════════════════════════════
# 10. Combined special/reserved token validation
# ══════════════════════════════════════════════════════════════════


def test_special_reserved_id_collision_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate token ID 0"):
        _validate_special_and_reserved_tokens({"<a>": 0}, {"<b>": 0})


def test_special_reserved_string_collision_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate token string"):
        _validate_special_and_reserved_tokens({"<a>": 0}, {"<a>": 1})


def test_boolean_special_id_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        _validate_special_and_reserved_tokens({"<a>": True}, {})


def test_special_id_collides_with_byte(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"text": "hello"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate token ID|byte IDs collide"):
        train_bpe(path, vocab_size=260, special_tokens={"<x>": 4})


# ══════════════════════════════════════════════════════════════════
# 11. Decode with skip_special_tokens
# ══════════════════════════════════════════════════════════════════


def test_decode_default_preserves_special(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    ids = [t.special_tokens["<bos>"], t.byte_value_to_id[ord("a")], t.special_tokens["<eos>"]]
    decoded = t.decode(ids)
    assert "<bos>" in decoded
    assert "<eos>" in decoded


def test_decode_skip_special_omits_them(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    ids = [t.special_tokens["<bos>"], t.byte_value_to_id[ord("a")], t.special_tokens["<eos>"]]
    decoded = t.decode(ids, skip_special_tokens=True)
    assert "<bos>" not in decoded
    assert "<eos>" not in decoded
    assert decoded.strip() == "a"


def test_decode_mixed_special_and_text(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    ids = [t.byte_value_to_id[ord("h")], t.special_tokens["<pad>"], t.byte_value_to_id[ord("i")]]
    decoded_default = t.decode(ids)
    assert "h" in decoded_default and "i" in decoded_default and "<pad>" in decoded_default


def test_decode_skip_special_multiple(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    ids = [
        t.special_tokens["<bos>"],
        t.byte_value_to_id[ord("a")],
        t.special_tokens["<pad>"],
        t.byte_value_to_id[ord("b")],
        t.special_tokens["<eos>"],
    ]
    decoded = t.decode(ids, skip_special_tokens=True)
    assert decoded == "ab"


# ══════════════════════════════════════════════════════════════════
# 12. Artifact validation
# ══════════════════════════════════════════════════════════════════


def test_validate_byte_id_range(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text('{"text": "hello"}\n', encoding="utf-8")
    t = train_bpe(path, vocab_size=260)
    t.validate()


def test_validate_catches_missing_byte(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    del t.byte_value_to_id[0]
    with pytest.raises(ValueError, match="must have exactly 256"):
        t.validate()


def test_validate_catches_byte_id_collision(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    t.byte_value_to_id[0] = t.special_tokens["<pad>"]
    with pytest.raises(ValueError, match="byte IDs collide"):
        t.validate()


def test_validate_catches_bad_merge_chain(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    if t.merges:
        bad_merge = dataclasses.replace(t.merges[0], left=99999)
        object.__setattr__(t, "merges", (bad_merge,) + t.merges[1:])
        with pytest.raises(ValueError, match="not in vocabulary"):
            t.validate()


def test_validate_hash_tampered(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["tokenizer_hash"] = "0" * 64
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        BPETokenizer.load(p)


# ══════════════════════════════════════════════════════════════════
# 13. Vocabulary behavior
# ══════════════════════════════════════════════════════════════════


def test_base_vocab_size_produces_no_merges(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    assert len(t.merges) == 0


def test_larger_vocab_produces_expected_merges(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=270)
    assert len(t.merges) == 10


def test_insufficient_pairs_reports_actual_size(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=10000)
    assert t.vocab_size < 10000
    assert t.vocab_size > 260


def test_merge_tokens_emitted_by_encode(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    ids = t.encode("hello world")
    merge_ids = {m.token for m in t.merges}
    used_merge_ids = merge_ids & set(ids)
    assert len(used_merge_ids) > 0


def test_decode_reconstructs_merge_payload(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    for m in t.merges:
        left_bytes = t.id_to_bytes[m.left]
        right_bytes = t.id_to_bytes[m.right]
        merge_bytes = t.id_to_bytes[m.token]
        assert merge_bytes == left_bytes + right_bytes


def test_all_256_bytes_reachable_after_save_load(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    t2 = BPETokenizer.load(p)
    for b in range(256):
        assert b in t2.byte_value_to_id
        tid = t2.byte_value_to_id[b]
        assert tid in t2.id_to_bytes
        assert t2.id_to_bytes[tid] == bytes([b])


def test_no_unk_for_valid_unicode(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    texts = ["hello", "नमस्ते", "😀", "日本語", "中文", "Español", "Français"]
    for text in texts:
        ids = t.encode(text)
        unk_id = t.special_tokens.get("<unk>", 1)
        assert unk_id not in ids, f"<unk> emitted for {text!r}"


# ══════════════════════════════════════════════════════════════════
# 14. Canonical serialization
# ══════════════════════════════════════════════════════════════════


def test_canonical_serialization_minified(corpus_en: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    serialized = t._compact_serialize()
    assert "\n" not in serialized
    assert "  " not in serialized


def test_hashed_vs_saved_consistent(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=280)
    h = t.compute_hash()
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    t2 = BPETokenizer.load(p)
    assert t2.compute_hash() == h


# ══════════════════════════════════════════════════════════════════
# 15. No-overwrite publication
# ══════════════════════════════════════════════════════════════════


def test_save_no_overwrite_default(corpus_en: Path, tmp_path: Path) -> None:
    t = train_bpe(corpus_en, vocab_size=260)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    with pytest.raises(FileExistsError):
        t.save(p)


def test_save_failure_cleans_temp(corpus_en: Path, tmp_path: Path) -> None:
    import dataclasses

    t = train_bpe(corpus_en, vocab_size=260)
    p = tmp_path / "t.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    broken = dataclasses.replace(t, schema_version="bad")
    with pytest.raises(ValueError):
        broken.save(p, overwrite=True)
    assert not any(
        f.name.startswith(f".{p.name}") and f.name.endswith(".tmp") for f in p.parent.iterdir()
    )


def test_save_atomic_no_destruction_on_failure(corpus_en: Path, tmp_path: Path) -> None:
    import dataclasses

    t = train_bpe(corpus_en, vocab_size=260)
    p = tmp_path / "t.json"
    t.save(p, overwrite=True)
    original = p.read_bytes()
    broken = dataclasses.replace(t, schema_version="bad")
    with pytest.raises(ValueError):
        broken.save(p, overwrite=True)
    assert p.read_bytes() == original
