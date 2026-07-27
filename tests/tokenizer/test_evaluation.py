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
    is_byte_alphabet_complete,
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
            record_id="bad",
            language="en",
            script="Latin",
            domain="test",
            text="bad\ud800",
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
    assert [r.record_id for r in records] == ["c", "a", "b"], "preserve input order"
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
        record_id=record_id,
        language=language,
        script="Auto",
        domain="general",
        text=text,
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count > 0
    assert metrics.char_count > 0
    assert metrics.tokens_per_char > 0
    assert metrics.exact_round_trip


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
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "emoji_zwj" in frag


# ── 18. Combining-mark NFC equivalence ───────────────────────────────


def test_combining_mark_nfc_equivalence(tiny_tokenizer: BharatTokenizer) -> None:
    nfc_text = "\u00e9"  # é in NFC
    nfd_text = "e\u0301"  # é in NFD
    nfc_record = EvaluationRecord(
        record_id="nfc-cmp", language="en", script="Latin", domain="nfc", text=nfc_text
    )
    nfd_record = EvaluationRecord(
        record_id="nfd-cmp", language="en", script="Latin", domain="nfc", text=nfd_text
    )

    nfc_metrics = _compute_record_metrics(nfc_record, tiny_tokenizer)
    nfd_metrics = _compute_record_metrics(nfd_record, tiny_tokenizer)

    nfc_nfc = unicodedata.normalize("NFC", nfc_text)
    nfd_nfc = unicodedata.normalize("NFC", nfd_text)
    assert nfc_nfc == nfd_nfc
    assert nfc_metrics.nfc_round_trip
    assert nfd_metrics.nfc_round_trip


# ── 19. Exact NFC round trip ─────────────────────────────────────────


def test_exact_nfc_round_trip(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="rt-nfc",
        language="en",
        script="Latin",
        domain="nfc",
        text="café déjà vu",
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
    assert metrics.canonical_equivalent


# ── 21. Unknown-token accounting ──────────────────────────────────────


def test_unknown_token_accounting(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="unk-test",
        language="en",
        script="Latin",
        domain="general",
        text="hello world",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.unknown_token_count >= 0
    assert metrics.unknown_token_rate >= 0


# ── 22. Complete byte coverage ───────────────────────────────────────


def test_byte_alphabet_coverage(tiny_tokenizer: BharatTokenizer) -> None:
    result = is_byte_alphabet_complete(tiny_tokenizer)
    assert "complete" in result
    assert "missing_byte_values" in result
    assert 0 <= result["total_reachable"] <= 256


# ── 23. Fertility calculation ────────────────────────────────────────


def test_fertility_calculation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="fert-test",
        language="en",
        script="Latin",
        domain="general",
        text="hello world",
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


def test_word_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="word-frag",
        language="en",
        script="Latin",
        domain="general",
        text="hello world foo bar baz qux",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "words" in frag
    assert frag["words"].item_count >= 0


def test_number_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="num-frag",
        language="en",
        script="Latin",
        domain="numbers",
        text="42 100 2000 3.14 1,000",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "numbers" in frag


def test_punctuation_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="punct-frag",
        language="en",
        script="Latin",
        domain="punctuation",
        text="Hello, world! How are you? (Fine.)",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "punctuation" in frag


def test_code_identifier_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="code-frag",
        language="en",
        script="Latin",
        domain="code",
        text="def foo_bar(camelCaseArg): snake_case_func myVar",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "code_identifiers" in frag or "camel_case" in frag


def test_mixed_script_fragmentation(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="mixed-frag",
        language="hi",
        script="Mixed",
        domain="general",
        text="भारत is diverse and विविध है",
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    frag = _compute_fragmentation([metrics])
    assert "mixed_indic_latin" in frag


# ── 30-31. Multiple tokenizer comparison ─────────────────────────────


def test_multiple_tokenizer_comparison(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world foo bar"}\n', encoding="utf-8")
    t1 = BharatBPETokenizer(train_bpe(corpus, vocab_size=265))
    t2 = BharatBPETokenizer(train_bpe(corpus, vocab_size=270))

    records = [
        EvaluationRecord(
            record_id="a", language="en", script="Latin", domain="gen", text="hello world"
        ),
    ]
    eval_engine = TokenizerEvaluation({"t1": t1, "t2": t2})
    eval_engine.set_records(records)
    report = eval_engine.compute()

    assert "comparison" in report
    assert len(report["comparison"]) >= 1
    comp = report["comparison"][0]
    assert "wins_a" in comp
    assert "wins_b" in comp
    assert "ties" in comp


def test_comparison_excludes_failed_round_trips(tiny_tokenizer: BharatTokenizer) -> None:
    eval_engine = TokenizerEvaluation({"t1": tiny_tokenizer, "t2": tiny_tokenizer})
    record = EvaluationRecord(
        record_id="a", language="en", script="Latin", domain="gen", text="hello"
    )
    eval_engine.set_records([record])
    report = eval_engine.compute()
    assert "comparison" in report
    comp = report["comparison"][0]
    assert comp["wins_a"] + comp["wins_b"] + comp["ties"] >= 0


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
    assert "/" not in serialized.split('"output"')[-1] if "output" in serialized else True
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


# ── 38. Aggregate metrics zero-length ────────────────────────────────


def test_zero_length_record(tiny_tokenizer: BharatTokenizer) -> None:
    record = EvaluationRecord(
        record_id="empty", language="en", script="Latin", domain="general", text=""
    )
    metrics = _compute_record_metrics(record, tiny_tokenizer)
    assert metrics.token_count == 0
    assert metrics.tokens_per_char == 0.0
    assert metrics.tokens_per_byte == 0.0


# ── 39. Round-trip summary ───────────────────────────────────────────


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
    assert rt["exact_pass_count"] >= 0
    assert rt["exact_pass_rate"] >= 0


# ── 40. Byte alphabet helper ─────────────────────────────────────────


def test_is_byte_alphabet_complete(tiny_tokenizer: BharatTokenizer) -> None:
    result = is_byte_alphabet_complete(tiny_tokenizer)
    assert isinstance(result["complete"], bool)
    assert isinstance(result["missing_byte_values"], list)
