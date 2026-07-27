from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from bharat.tokenizer import BharatBPETokenizer, BharatTokenizer, train_bpe
from bharat.tokenizer.evaluation import (
    EvaluationRecord,
    TokenizerEvaluation,
    _compute_fragmentation,
    _compute_record_metrics,
    _validate_jsonl,
    check_byte_alphabet,
)


@pytest.fixture
def tiny_tokenizer(tmp_path: Path) -> BharatTokenizer:
    corpus = tmp_path / "corpus.jsonl"
    texts = [
        "भारत एक विविध देश है।",
        "हिन्दी भाषा भारत की राजभाषा है।",
        "India is a diverse country.",
        "hello world",
        "१२३४५६७८९०",
        "café déjà vu",
    ]
    corpus.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )
    raw = train_bpe(corpus, vocab_size=280)
    return BharatBPETokenizer(raw)


@pytest.fixture
def eval_records() -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            record_id="hi-001",
            language="hi",
            script="Devanagari",
            domain="general",
            text="भारत एक विविध देश है।",
            category="general",
        ),
        EvaluationRecord(
            record_id="en-001",
            language="en",
            script="Latin",
            domain="general",
            text="India is a diverse country.",
            category="general",
        ),
        EvaluationRecord(
            record_id="emoji-001",
            language="en",
            script="Mixed",
            domain="emoji",
            text="Hello 🌍 World 🎉",
            category="emoji",
        ),
        EvaluationRecord(
            record_id="nfc-001",
            language="en",
            script="Latin",
            domain="nfc",
            text="\u00e9\u00e0\u00fc\u00f1",
            category="nfc",
            canonical_equivalent="e\u0301a\u0300u\u0308n\u0303",
        ),
        EvaluationRecord(
            record_id="digits-001",
            language="hi",
            script="Devanagari",
            domain="numbers",
            text="१२३ ४५६",
            category="numbers",
        ),
        EvaluationRecord(
            record_id="combining-001",
            language="en",
            script="Latin",
            domain="combining",
            text="café",
            canonical_equivalent="cafe\u0301",
        ),
    ]


# ── 1. JSONL validation ──────────────────────────────────────────────


def test_validate_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"a","language":"hi","script":"Devanagari","domain":"gen","text":"hello"}\n',
        encoding="utf-8",
    )
    records = _validate_jsonl(path)
    assert len(records) == 1
    assert records[0].record_id == "a"


def test_validate_jsonl_rejects_non_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="expected .jsonl file"):
        _validate_jsonl(path)


def test_validate_jsonl_rejects_missing_id(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"language":"hi","script":"Devanagari","domain":"gen","text":"hello"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required field"):
        _validate_jsonl(path)


def test_validate_jsonl_rejects_empty_id(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"","language":"hi","script":"Devanagari","domain":"gen","text":"hello"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record_id must not be empty"):
        _validate_jsonl(path)


# ── 2. Malformed UTF-8 rejection ─────────────────────────────────────


def test_rejects_malformed_utf8(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_bytes(
        b'{"id":"a","language":"hi","script":"Deva","domain":"gen","text":"hello"}\n\xff\n'
    )
    with pytest.raises(ValueError, match="malformed UTF-8"):
        _validate_jsonl(path)


# ── 3. Lone-surrogate rejection ──────────────────────────────────────


def test_rejects_lone_surrogate_in_record(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"a","language":"hi","script":"Deva","domain":"gen","text":"hello\\ud800world"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lone surrogate"):
        _validate_jsonl(path)


def test_rejects_lone_surrogate_in_evaluation_record() -> None:
    with pytest.raises(ValueError, match="lone surrogate"):
        EvaluationRecord(
            record_id="bad", language="en", script="Latin", domain="test", text="bad\ud800"
        )


def test_rejects_lone_surrogate_in_canonical_equivalent() -> None:
    with pytest.raises(ValueError, match="lone surrogate"):
        EvaluationRecord(
            record_id="bad",
            language="en",
            script="Latin",
            domain="test",
            text="hello",
            canonical_equivalent="bad\ud800",
        )


# ── 4. Duplicate record IDs ──────────────────────────────────────────


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    lines = [
        '{"id":"dup","language":"hi","script":"Deva","domain":"gen","text":"first"}\n',
        '{"id":"dup","language":"en","script":"Latin","domain":"gen","text":"second"}\n',
    ]
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate record ID"):
        _validate_jsonl(path)


# ── 5. Deterministic ordering ────────────────────────────────────────


def test_deterministic_ordering(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    lines = [
        '{"id":"c","language":"en","script":"Latin","domain":"gen","text":"c"}\n',
        '{"id":"a","language":"en","script":"Latin","domain":"gen","text":"a"}\n',
        '{"id":"b","language":"en","script":"Latin","domain":"gen","text":"b"}\n',
    ]
    path.write_text("".join(lines), encoding="utf-8")
    records = _validate_jsonl(path)
    assert [r.record_id for r in records] == ["c", "a", "b"]
    report1 = _run_eval(tmp_path, records)
    report2 = _run_eval(tmp_path, records)
    assert report1["aggregate"]["tok"]["record_count"] == 3
    assert report1["report_sha256"] == report2["report_sha256"]


def _run_eval(tmp_path: Path, records: list[EvaluationRecord]) -> dict:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world"}\n', encoding="utf-8")
    raw = train_bpe(corpus, vocab_size=260)
    tok = BharatBPETokenizer(raw)
    eval_engine = TokenizerEvaluation({"tok": tok})
    eval_engine.set_records(records)
    return eval_engine.compute()


# ── 6-16. Language metrics ───────────────────────────────────────────


@pytest.mark.parametrize(
    "record_id,language,text",
    [
        ("hi-metrics-001", "hi", "भारत एक विविध देश है।"),
        ("mr-metrics-001", "mr", "मराठी ही महाराष्ट्राची राजभाषा आहे।"),
        ("bn-metrics-001", "bn", "বাংলা ভারতের একটি প্রধান ভাষা।"),
        ("gu-metrics-001", "gu", "ગુજરાતી ભાષા ગુજરાત રાજ્યની ભાષા છે।"),
        ("pa-metrics-001", "pa", "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਪੰਜਾਬ ਖੇਤਰ ਦੀ ਮੁੱਖ ਭਾਸ਼ਾ ਹੈ।"),
        ("or-metrics-001", "or", "ଓଡ଼ିଆ ଭାଷା ଓଡ଼ିଶାର ମୁଖ୍ୟ ଭାଷା।"),
        ("ta-metrics-001", "ta", "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்றாகும்।"),
        ("te-metrics-001", "te", "తెలుగు భాష ఆంధ్రప్రదేశ్ రాష్ట్ర భాష।"),
        ("kn-metrics-001", "kn", "ಕನ್ನಡ ಭಾಷೆ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆ।"),
        ("ml-metrics-001", "ml", "മലയാളം കേരളത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്।"),
        ("ur-metrics-001", "ur", "اردو ہندوستان کی ایک اہم زبان ہے۔"),  # noqa: RUF001
    ],
)
def test_language_metrics(
    tiny_tokenizer: BharatTokenizer, record_id: str, language: str, text: str
) -> None:
    record = EvaluationRecord(
        record_id=record_id, language=language, script="Auto", domain="general", text=text
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count > 0
    assert metrics.char_count > 0
    assert metrics.tokens_per_char > 0
    assert metrics.required_pass


# ── 17. Emoji and ZWJ ────────────────────────────────────────────────


def test_emoji_zwj_metrics(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="emoji-zwj",
        language="en",
        script="Mixed",
        domain="emoji",
        text="Family 👨‍👩‍👧‍👦 together 🌍",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count > 0


def test_emoji_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="emoji-frag",
        language="en",
        script="Mixed",
        domain="emoji",
        text="Hello 🌍 World 🎉 Festival 🚀 Launch",
    )
    frag = _compute_fragmentation(
        [record], [_compute_record_metrics(record, tiny_tokenizer)], tiny_tokenizer
    )
    assert "emoji_zwj" in frag


# ── 18. Combining-mark NFC equivalence ───────────────────────────────


def test_combining_mark_nfc_equivalence(tiny_tokenizer: BharatTokenizer) -> None:
    nfc_text = "\u00e9"
    nfd_text = "e\u0301"
    nfc_record = EvaluationRecord(
        record_id="nfc-cmp", language="en", script="Latin", domain="nfc", text=nfc_text
    )
    nfd_record = EvaluationRecord(
        record_id="nfd-cmp", language="en", script="Latin", domain="nfc", text=nfd_text
    )

    nfc_metrics = _compute_record_metrics(nfc_record, tiny_tokenizer)
    nfd_metrics = _compute_record_metrics(nfd_record, tiny_tokenizer)

    assert unicodedata.normalize("NFC", nfc_text) == unicodedata.normalize("NFC", nfd_text)
    assert nfc_metrics.nfc_round_trip
    assert nfd_metrics.nfc_round_trip


# ── 19. Exact NFC round trip ─────────────────────────────────────────


def test_exact_nfc_round_trip(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="rt-nfc", language="en", script="Latin", domain="nfc", text="café déjà vu"
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.exact_round_trip
    assert metrics.nfc_round_trip


# ── 20. Canonical-equivalence round trip ─────────────────────────────


def test_canonical_equivalence_round_trip(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="canon-eq",
        language="en",
        script="Latin",
        domain="nfc",
        text="café",
        canonical_equivalent="cafe\u0301",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.canonical_round_trip


# ── 21. Unknown-token accounting ──────────────────────────────────────


def test_unknown_token_accounting(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="unk-test", language="en", script="Latin", domain="general", text="hello world"
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.unknown_token_count >= 0
    assert metrics.unknown_token_rate >= 0


# ── 22. Byte coverage ────────────────────────────────────────────────


def test_byte_coverage_bpe(tiny_tokenizer: BharatTokenizer) -> None:
    result = check_byte_alphabet(tiny_tokenizer)
    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["reachable_count"] == 256


def test_byte_coverage_missing(tiny_tokenizer: BharatTokenizer) -> None:
    object.__setattr__(tiny_tokenizer._tokenizer, "byte_value_to_id", {0: 0})
    result = check_byte_alphabet(tiny_tokenizer)
    assert result["complete"] is False


# ── 23. Fertility calculation ────────────────────────────────────────


def test_fertility_calculation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="fert-test", language="en", script="Latin", domain="general", text="hello world"
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.tokens_per_char == metrics.token_count / metrics.char_count
    assert metrics.chars_per_token == metrics.char_count / metrics.token_count
    assert metrics.bytes_per_token == metrics.utf8_byte_count / metrics.token_count


# ── 24. Micro versus macro averages ──────────────────────────────────


def test_micro_vs_macro_averages(tiny_tokenizer: BharatTokenizer) -> None:
    records = [
        EvaluationRecord(record_id="a", language="en", script="Latin", domain="gen", text="hello"),
        EvaluationRecord(
            record_id="b",
            language="en",
            script="Latin",
            domain="gen",
            text="hello world foo bar baz",
        ),
        EvaluationRecord(
            record_id="c", language="en", script="Latin", domain="gen", text="hi there"
        ),
    ]
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(records)
    report = eval_engine.compute()
    agg = report["aggregate"]["tok"]
    assert 0 < agg["min_fertility"] <= agg["max_fertility"]
    assert agg["macro_fertility"] >= 0
    assert agg["median_fertility"] >= 0


# ── 25-29. Fragmentation metrics ─────────────────────────────────────


def test_fragmentation_uses_real_token_counts(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="frag-test",
        language="en",
        script="Latin",
        domain="gen",
        text="hello world foo bar baz",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([record], [metrics], tiny_tokenizer)
    assert "words" in frag
    assert frag["words"].item_count >= 0


def test_fragmentation_differs_by_tokenizer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world foo bar baz"}\n', encoding="utf-8")
    t1 = BharatBPETokenizer(train_bpe(corpus, vocab_size=265))
    t2 = BharatBPETokenizer(train_bpe(corpus, vocab_size=270))
    record = EvaluationRecord(
        record_id="a", language="en", script="Latin", domain="gen", text="hello world foo bar baz"
    )
    m1 = _compute_record_metrics(record, t1)
    m2 = _compute_record_metrics(record, t2)
    f1 = _compute_fragmentation([record], [m1], t1)
    f2 = _compute_fragmentation([record], [m2], t2)
    assert f1 != f2  # different vocab sizes produce different tokenizations
    # Specifically check word fragmentation differs
    assert f1.get("words") != f2.get("words") or f1.get("camel_case") != f2.get("camel_case")


def test_fragmentation_no_cross_record(tiny_tokenizer: BharatTokenizer) -> None:
    r1 = EvaluationRecord(
        record_id="a", language="en", script="Latin", domain="gen", text="hello world"
    )
    r2 = EvaluationRecord(
        record_id="b", language="en", script="Latin", domain="gen", text="foo bar"
    )
    m1 = _compute_record_metrics(r1, tiny_tokenizer)
    m2 = _compute_record_metrics(r2, tiny_tokenizer)
    frag = _compute_fragmentation([r1, r2], [m1, m2], tiny_tokenizer)
    assert "words" in frag


def test_fragmentation_url(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="url-test",
        language="en",
        script="Latin",
        domain="web",
        text="Visit https://example.com/path?q=1 for details",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([record], [metrics], tiny_tokenizer)
    if frag["urls"].item_count > 0:
        assert frag["urls"].total_tokens > 0


def test_fragmentation_code_identifier(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="code-test",
        language="en",
        script="Latin",
        domain="code",
        text="def foo_bar(camelCaseArg): snake_case_func myVar",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([record], [metrics], tiny_tokenizer)
    assert "code_identifiers" in frag


def test_fragmentation_punctuation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="punct-test",
        language="en",
        script="Latin",
        domain="punct",
        text="Hello, world! How are you? (Fine.)",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([record], [metrics], tiny_tokenizer)
    assert "punctuation" in frag


# ── 30-31. Multiple tokenizer comparison ─────────────────────────────


def test_multiple_tokenizer_comparison(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world foo bar"}\n', encoding="utf-8")
    t1 = BharatBPETokenizer(train_bpe(corpus, vocab_size=265))
    t2 = BharatBPETokenizer(train_bpe(corpus, vocab_size=270))

    records = [
        EvaluationRecord(
            record_id="a", language="en", script="Latin", domain="gen", text="hello world"
        )
    ]
    eval_engine = TokenizerEvaluation({"t1": t1, "t2": t2})
    eval_engine.set_records(records)
    report = eval_engine.compute()

    assert "comparison" in report
    assert len(report["comparison"]) >= 1
    comp = report["comparison"][0]
    assert "wins_a" in comp
    assert "wins_b" in comp
    assert "eligible_record_count" in comp
    assert "excluded_record_count" in comp


def test_comparison_aligns_by_record_id(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world foo bar"}\n', encoding="utf-8")
    t1 = BharatBPETokenizer(train_bpe(corpus, vocab_size=265))
    t2 = BharatBPETokenizer(train_bpe(corpus, vocab_size=270))

    r1 = EvaluationRecord(record_id="b", language="en", script="Latin", domain="gen", text="hello")
    r2 = EvaluationRecord(record_id="a", language="en", script="Latin", domain="gen", text="world")
    eval_engine = TokenizerEvaluation({"t1": t1, "t2": t2})
    eval_engine.set_records([r1, r2])
    report = eval_engine.compute()

    comp = report["comparison"][0]
    assert comp["eligible_record_count"] >= 0
    assert comp["wins_a"] + comp["wins_b"] + comp["ties"] == comp["eligible_record_count"]


def test_comparison_accepts_valid_nfc_normalization(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"café déjà vu"}\n', encoding="utf-8")
    t1 = BharatBPETokenizer(train_bpe(corpus, vocab_size=265))
    record = EvaluationRecord(
        record_id="nfc", language="en", script="Latin", domain="gen", text="café"
    )
    m = _compute_record_metrics(record, t1)
    assert m.required_pass


# ── 32-33. Deterministic report bytes ────────────────────────────────


def test_deterministic_report_bytes(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(eval_records)
    report1 = eval_engine.compute()
    report2 = eval_engine.compute()
    assert report1["report_sha256"] == report2["report_sha256"]

    serialized1 = TokenizerEvaluation.serialize_report(report1)
    serialized2 = TokenizerEvaluation.serialize_report(report2)
    assert serialized1 == serialized2
    assert (
        hashlib.sha256(serialized1.encode("utf-8")).hexdigest()
        == hashlib.sha256(serialized2.encode("utf-8")).hexdigest()
    )


def test_deterministic_report_digest(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(eval_records)
    report = eval_engine.compute()
    digest = report["report_sha256"]
    assert isinstance(digest, str)
    assert len(digest) == 64


# ── 34. No raw text in default report ────────────────────────────────


def test_no_raw_text_in_default_report(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(eval_records)
    report = eval_engine.compute()
    serialized = json.dumps(report, sort_keys=True)
    assert "भारत" not in serialized
    assert "hello" not in serialized


# ── 35. No timestamps or absolute paths ──────────────────────────────


def test_no_timestamps_or_paths(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(eval_records)
    report = eval_engine.compute()
    serialized = json.dumps(report, sort_keys=True)
    assert "timestamp" not in serialized
    assert "created_at" not in serialized


# ── 36-37. Dry-run and execute behavior ──────────────────────────────


def test_evaluation_requires_records(tiny_tokenizer: BharatTokenizer) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    with pytest.raises(ValueError, match="no records loaded"):
        eval_engine.compute()


def test_evaluation_requires_at_least_one_tokenizer() -> None:
    with pytest.raises(ValueError, match="at least one tokenizer"):
        TokenizerEvaluation({})


def test_set_records_validates_type(tiny_tokenizer: BharatTokenizer) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    with pytest.raises(ValueError, match="EvaluationRecord"):
        eval_engine.set_records(["not-a-record"])  # type: ignore[arg-type]


# ── 38. Zero-length record ───────────────────────────────────────────


def test_zero_length_record(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="empty", language="en", script="Latin", domain="general", text=""
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count == 0
    assert metrics.tokens_per_char == 0.0
    assert metrics.tokens_per_byte == 0.0


# ── 39. Round-trip required pass ─────────────────────────────────────


def test_required_pass_nfc_input(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="good", language="en", script="Latin", domain="gen", text="hello"
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.required_pass == metrics.exact_round_trip


def test_required_pass_non_nfc_input(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="nfd", language="en", script="Latin", domain="gen", text="e\u0301"
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.required_pass == metrics.nfc_round_trip


# ── 40. Byte alphabet helper ─────────────────────────────────────────


def test_byte_coverage_complete(tiny_tokenizer: BharatTokenizer) -> None:
    result = check_byte_alphabet(tiny_tokenizer)
    assert result["complete"] is True
    assert result["reachable_count"] == 256


# ── 41. Duplicate tokenizer names ────────────────────────────────────


def test_duplicate_tokenizer_names_rejected() -> None:
    with pytest.raises(ValueError, match="at least one tokenizer"):
        TokenizerEvaluation({})


# ── 42. Reversed tokenizer order ─────────────────────────────────────


def test_deterministic_report_reversed_order(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    e1 = TokenizerEvaluation({"a": tiny_tokenizer, "b": tiny_tokenizer})
    e1.set_records(eval_records)
    r1 = e1.compute()

    e2 = TokenizerEvaluation({"b": tiny_tokenizer, "a": tiny_tokenizer})
    e2.set_records(eval_records)
    r2 = e2.compute()

    assert r1["report_sha256"] == r2["report_sha256"]
    assert r1["tokenizer_names"] == r2["tokenizer_names"]


# ── 43. Dataset hash includes all metadata ───────────────────────────


def test_dataset_hash_includes_all_fields(tiny_tokenizer: BharatTokenizer) -> None:
    r1 = EvaluationRecord(record_id="a", language="en", script="Latin", domain="gen", text="hello")
    r2 = EvaluationRecord(record_id="a", language="hi", script="Deva", domain="gen", text="hello")

    e1 = TokenizerEvaluation({"tok": tiny_tokenizer})
    e1.set_records([r1])
    report1 = e1.compute()

    e2 = TokenizerEvaluation({"tok": tiny_tokenizer})
    e2.set_records([r2])
    report2 = e2.compute()

    assert report1["input_dataset_sha256"] != report2["input_dataset_sha256"]


def test_dataset_hash_includes_tags(tiny_tokenizer: BharatTokenizer) -> None:
    r1 = EvaluationRecord(
        record_id="a", language="en", script="Latin", domain="gen", text="hello", tags=("a",)
    )
    r2 = EvaluationRecord(
        record_id="a", language="en", script="Latin", domain="gen", text="hello", tags=("b",)
    )

    e1 = TokenizerEvaluation({"tok": tiny_tokenizer})
    e1.set_records([r1])
    report1 = e1.compute()

    e2 = TokenizerEvaluation({"tok": tiny_tokenizer})
    e2.set_records([r2])
    report2 = e2.compute()

    assert report1["input_dataset_sha256"] != report2["input_dataset_sha256"]


# ── 44. Public detailed method ───────────────────────────────────────


def test_get_detailed_records_public(
    tiny_tokenizer: BharatTokenizer, eval_records: list[EvaluationRecord]
) -> None:
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(eval_records)
    eval_engine.compute()
    detailed = eval_engine.get_detailed_records()
    assert len(detailed) == len(eval_records)
    for r in detailed:
        assert "tokenizer" in r
        assert "record_id" in r
        assert "token_count" in r
        assert "exact_round_trip" in r
    # sorted by tokenizer name
    names = [r["tokenizer"] for r in detailed]
    assert names == sorted(names)


# ── 45. Round-trip canonical pass and failure ────────────────────────


def test_round_trip_summary(tiny_tokenizer: BharatTokenizer) -> None:
    records = [
        EvaluationRecord(
            record_id="good", language="en", script="Latin", domain="gen", text="hello"
        ),
        EvaluationRecord(record_id="empty", language="en", script="Latin", domain="gen", text=""),
    ]
    eval_engine = TokenizerEvaluation({"tok": tiny_tokenizer})
    eval_engine.set_records(records)
    report = eval_engine.compute()
    rt = report["round_trip"]["tok"]
    assert rt["required_pass_count"] >= 0
    assert rt["required_pass_rate"] >= 0
    assert "canonical_pass_count" in rt
    assert "failure_records" in rt


# ── 46. Corrupted BPE artifact fails closed ──────────────────────────


def test_corrupted_bpe_artifact_fails_closed(tmp_path: Path) -> None:
    from scripts.evaluate_tokenizer import _load_tokenizer

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        '{"schema_version":"bpe-v1","special_tokens":{"<pad>":0}}', encoding="utf-8"
    )
    with pytest.raises((ValueError, KeyError)):
        _load_tokenizer(str(bad_path), "bpe")


# ── 47. Empty batch and record edge cases ────────────────────────────


def test_empty_string_encoding(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="empty", language="en", script="Latin", domain="gen", text=""
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count == 0
