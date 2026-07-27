from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final, Protocol

from bharat.tokenizer.base import BharatTokenizer

_SCHEMA_VERSION = "eval-v1"
_EVALUATOR_VERSION = "1.0.3"

_RE_WORD = re.compile(r"\w+", re.UNICODE)
_RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_RE_PUNCTUATION = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]+")
_RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
_RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_CODE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*|[a-z][a-zA-Z0-9]*|[A-Z][a-z0-9]+[a-zA-Z0-9]*")
_RE_CAMEL_CASE = re.compile(r"[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)")
_RE_SNAKE_CASE = re.compile(r"[a-z]+(?:_[a-z]+)*")
_RE_MIXED_INDIC_LATIN = re.compile(r"[\u0080-\uFFFF]*[a-zA-Z]+[\u0080-\uFFFF]*")
_RE_EMOJI_ZWJ = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "\u200d"
    "\ufe0f"
    "]+"
)

_FRAGMENTATION_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "words": _RE_WORD,
    "numbers": _RE_NUMBER,
    "punctuation": _RE_PUNCTUATION,
    "urls": _RE_URL,
    "emails": _RE_EMAIL,
    "hashtags": _RE_HASHTAG,
    "code_identifiers": _RE_CODE_IDENTIFIER,
    "camel_case": _RE_CAMEL_CASE,
    "snake_case": _RE_SNAKE_CASE,
    "mixed_indic_latin": _RE_MIXED_INDIC_LATIN,
    "emoji_zwj": _RE_EMOJI_ZWJ,
}


# ── Byte-alphabet protocol ───────────────────────────────────────────


class ByteAlphabetProvider(Protocol):
    """Optional capability: expose the full byte-to-ID mapping."""

    @property
    def byte_value_to_id(self) -> dict[int, int]: ...


def check_byte_alphabet(tokenizer: BharatTokenizer) -> dict[str, Any]:
    """Return truthful byte-coverage data.

    For BharatBPETokenizer (which exposes byte_value_to_id) this verifies
    exactly 256 entries, keys 0-255, unique IDs, no collision with
    special/reserved, and each ID mapping to the correct single-byte payload.
    For generic tokenizers without the attribute the result is
    ``status: "unavailable"``.
    """
    provider = getattr(tokenizer, "byte_value_to_id", None)
    if provider is None:
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None:
            provider = getattr(inner, "byte_value_to_id", None)
    if provider is None:
        return {
            "status": "unavailable",
            "complete": False,
            "reachable_count": 0,
            "missing_byte_values": [],
        }

    mapping: dict[int, int] = provider
    if len(mapping) != 256:
        return {
            "status": "incomplete",
            "complete": False,
            "reachable_count": len(mapping),
            "missing_byte_values": sorted(set(range(256)) - set(mapping)),
        }

    if set(mapping) != set(range(256)):
        return {
            "status": "incomplete",
            "complete": False,
            "reachable_count": len(mapping),
            "missing_byte_values": sorted(set(range(256)) - set(mapping)),
        }

    seen: set[int] = set()
    for _byte_val, tid in sorted(mapping.items()):
        if tid in seen:
            return {
                "status": "incomplete",
                "complete": False,
                "reachable_count": len(mapping),
                "missing_byte_values": [],
            }
        seen.add(tid)

    id_to_bytes = getattr(tokenizer, "id_to_bytes", None)
    if id_to_bytes is None:
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None:
            id_to_bytes = getattr(inner, "id_to_bytes", None)
    if id_to_bytes is not None:
        for byte_val2, tid2 in sorted(mapping.items()):
            expected = bytes([byte_val2])
            if id_to_bytes.get(tid2) != expected:
                return {
                    "status": "incomplete",
                    "complete": False,
                    "reachable_count": len(mapping),
                    "missing_byte_values": [],
                }

    return {
        "status": "complete",
        "complete": True,
        "reachable_count": 256,
        "missing_byte_values": [],
    }


# ── Evaluation record ────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    language: str
    script: str
    domain: str
    text: str
    tags: tuple[str, ...] = ()
    canonical_equivalent: str | None = None
    category: str = "general"

    def __post_init__(self) -> None:
        if not self.record_id:
            msg = "record_id must not be empty"
            raise ValueError(msg)
        if not self.language:
            msg = "language must not be empty"
            raise ValueError(msg)
        if not isinstance(self.text, str):
            msg = f"text must be a string, got {type(self.text).__name__}"
            raise ValueError(msg)
        self._check_surrogates(self.text, "text")
        if self.canonical_equivalent is not None:
            if not isinstance(self.canonical_equivalent, str):
                msg = "canonical_equivalent must be a string"
                raise ValueError(msg)
            self._check_surrogates(self.canonical_equivalent, "canonical_equivalent")

    @staticmethod
    def _check_surrogates(text: str, field: str) -> None:
        for surrogate in re.findall(r"[\ud800-\udfff]", text):
            msg = f"lone surrogate U+{ord(surrogate):04X} in {field}"
            raise ValueError(msg)


# ── JSONL validation ────────────────────────────────────────────────


def _validate_jsonl(path: Path) -> list[EvaluationRecord]:
    if path.suffix != ".jsonl":
        msg = f"expected .jsonl file, got {path.suffix}"
        raise ValueError(msg)

    records: list[EvaluationRecord] = []
    seen_ids: set[str] = set()

    for line_num, raw_line in enumerate(path.read_bytes().split(b"\n"), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            decoded = stripped.decode("utf-8")
        except UnicodeDecodeError as e:
            msg = f"line {line_num}: malformed UTF-8"
            raise ValueError(msg) from e

        try:
            obj = json.loads(decoded)
        except json.JSONDecodeError as e:
            msg = f"line {line_num}: malformed JSON"
            raise ValueError(msg) from e

        if not isinstance(obj, dict):
            msg = f"line {line_num}: expected JSON object, got {type(obj).__name__}"
            raise ValueError(msg)

        for key in ("id", "language", "script", "domain", "text"):
            if key not in obj:
                msg = f"line {line_num}: missing required field {key!r}"
                raise ValueError(msg)
            if not isinstance(obj[key], str):
                msg = f"line {line_num}: field {key!r} must be a string"
                raise ValueError(msg)

        record_id = obj["id"]
        if record_id in seen_ids:
            msg = f"line {line_num}: duplicate record ID {record_id!r}"
            raise ValueError(msg)
        seen_ids.add(record_id)

        text = obj["text"]
        EvaluationRecord._check_surrogates(text, "text")

        tags_raw = obj.get("tags")
        if tags_raw is not None and (
            not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw)
        ):
            msg = f"line {line_num}: tags must be a list of strings"
            raise ValueError(msg)
        tags = tuple(tags_raw) if tags_raw else ()

        canonical = obj.get("canonical_equivalent")
        if canonical is not None:
            if not isinstance(canonical, str):
                msg = f"line {line_num}: canonical_equivalent must be a string"
                raise ValueError(msg)
            EvaluationRecord._check_surrogates(canonical, "canonical_equivalent")

        category = obj.get("category", "general")
        if not isinstance(category, str):
            msg = f"line {line_num}: category must be a string"
            raise ValueError(msg)

        records.append(
            EvaluationRecord(
                record_id=record_id,
                language=obj["language"],
                script=obj["script"],
                domain=obj["domain"],
                text=text,
                tags=tags,
                canonical_equivalent=canonical,
                category=category,
            )
        )

    return records


# ── Per-record metrics ──────────────────────────────────────────────


@dataclass(frozen=True)
class RecordMetrics:
    record_id: str
    language: str
    script: str
    domain: str
    category: str
    char_count: int
    codepoint_count: int
    utf8_byte_count: int
    token_count: int
    tokens_per_char: float
    tokens_per_byte: float
    chars_per_token: float
    bytes_per_token: float
    unknown_token_count: int
    unknown_token_rate: float
    special_token_count: int
    byte_token_count: int | None
    merged_token_count: int | None
    decoded_text: str
    exact_round_trip: bool
    nfc_round_trip: bool
    canonical_round_trip: bool
    required_pass: bool


def _compute_record_metrics(
    record: EvaluationRecord,
    tokenizer: BharatTokenizer,
) -> RecordMetrics:
    text = record.text
    nfc_text = unicodedata.normalize("NFC", text)

    char_count = len(text)
    codepoint_count = len(text)
    utf8_byte_count = len(text.encode("utf-8"))

    ids = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(ids)

    decoded = tokenizer.decode(ids)

    tokens_per_char = token_count / char_count if char_count > 0 else 0.0
    tokens_per_byte = token_count / utf8_byte_count if utf8_byte_count > 0 else 0.0
    chars_per_token = char_count / token_count if token_count > 0 else 0.0
    bytes_per_token = utf8_byte_count / token_count if token_count > 0 else 0.0

    special_ids: set[int] = set()
    for attr in ("pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
        try:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)
        except (ValueError, NotImplementedError):
            pass

    special_token_count = sum(1 for tid in ids if tid in special_ids)
    unknown_token_count = 0
    try:
        unk_id = tokenizer.unk_token_id
        unknown_token_count = sum(1 for tid in ids if tid == unk_id)
    except (ValueError, NotImplementedError):
        pass

    unknown_token_rate = unknown_token_count / token_count if token_count > 0 else 0.0

    byte_token_count: int | None = None
    merged_token_count: int | None = None
    tokenizer_meta = getattr(tokenizer, "get_metadata", None)
    if tokenizer_meta is not None:
        meta = tokenizer_meta()
        base_size = meta.get("base_vocab_size")
        if base_size is not None:
            byte_token_count = sum(1 for tid in ids if tid < base_size and tid not in special_ids)
            merged_token_count = sum(1 for tid in ids if tid >= base_size)

    exact_round_trip = decoded == text
    nfc_round_trip = decoded == nfc_text
    canonical_round_trip = True
    if record.canonical_equivalent is not None:
        canonical_ids = tokenizer.encode(record.canonical_equivalent, add_special_tokens=False)
        canonical_decoded = tokenizer.decode(canonical_ids)
        canonical_round_trip = canonical_decoded == nfc_text

    required_pass = exact_round_trip if text == nfc_text else nfc_round_trip

    return RecordMetrics(
        record_id=record.record_id,
        language=record.language,
        script=record.script,
        domain=record.domain,
        category=record.category,
        char_count=char_count,
        codepoint_count=codepoint_count,
        utf8_byte_count=utf8_byte_count,
        token_count=token_count,
        tokens_per_char=tokens_per_char,
        tokens_per_byte=tokens_per_byte,
        chars_per_token=chars_per_token,
        bytes_per_token=bytes_per_token,
        unknown_token_count=unknown_token_count,
        unknown_token_rate=unknown_token_rate,
        special_token_count=special_token_count,
        byte_token_count=byte_token_count,
        merged_token_count=merged_token_count,
        decoded_text=decoded,
        exact_round_trip=exact_round_trip,
        nfc_round_trip=nfc_round_trip,
        canonical_round_trip=canonical_round_trip,
        required_pass=required_pass,
    )


# ── Aggregates ──────────────────────────────────────────────────────


@dataclass
class AggregateMetrics:
    record_count: int
    char_count: int
    byte_count: int
    token_count: int
    micro_fertility: float
    macro_fertility: float
    min_fertility: float
    max_fertility: float
    median_fertility: float


def _compute_aggregate(metrics: list[RecordMetrics]) -> AggregateMetrics:
    record_count = len(metrics)
    char_count = sum(m.char_count for m in metrics)
    byte_count = sum(m.utf8_byte_count for m in metrics)
    token_count = sum(m.token_count for m in metrics)

    fertilities = [m.tokens_per_char for m in metrics]
    micro_fertility = token_count / char_count if char_count > 0 else 0.0
    macro_fertility = sum(fertilities) / record_count if record_count > 0 else 0.0
    min_fertility = min(fertilities) if fertilities else 0.0
    max_fertility = max(fertilities) if fertilities else 0.0
    median_fertility = median(fertilities) if fertilities else 0.0

    return AggregateMetrics(
        record_count=record_count,
        char_count=char_count,
        byte_count=byte_count,
        token_count=token_count,
        micro_fertility=micro_fertility,
        macro_fertility=macro_fertility,
        min_fertility=min_fertility,
        max_fertility=max_fertility,
        median_fertility=median_fertility,
    )


# ── Round-trip reporting ────────────────────────────────────────────


@dataclass
class RoundTripSummary:
    exact_pass_count: int
    exact_pass_rate: float
    nfc_pass_count: int
    nfc_pass_rate: float
    canonical_pass_count: int
    canonical_pass_rate: float
    required_pass_count: int
    required_pass_rate: float
    failure_records: list[dict[str, str]]


def _compute_round_trip(
    metrics: list[RecordMetrics], records: list[EvaluationRecord]
) -> RoundTripSummary:
    total = len(metrics)
    exact_pass = sum(1 for m in metrics if m.exact_round_trip)
    nfc_pass = sum(1 for m in metrics if m.nfc_round_trip)
    canonical_pass = sum(1 for m in metrics if m.canonical_round_trip)
    required_pass = sum(1 for m in metrics if m.required_pass)

    failure_records: list[dict[str, str]] = []
    by_id = {r.record_id: r for r in records}
    for m in metrics:
        reasons: list[str] = []
        if not m.exact_round_trip:
            reasons.append("exact")
        if not m.nfc_round_trip:
            reasons.append("nfc")
        if not m.canonical_round_trip:
            reasons.append("canonical")
        if not m.required_pass:
            reasons.append("required")
        if reasons:
            rec = by_id.get(m.record_id)
            failure_records.append(
                {
                    "record_id": m.record_id,
                    "failure_reasons": ",".join(reasons),
                    "is_nfc_input": str(
                        rec is not None and unicodedata.normalize("NFC", rec.text) == rec.text
                    ),
                }
            )

    return RoundTripSummary(
        exact_pass_count=exact_pass,
        exact_pass_rate=exact_pass / total if total > 0 else 1.0,
        nfc_pass_count=nfc_pass,
        nfc_pass_rate=nfc_pass / total if total > 0 else 1.0,
        canonical_pass_count=canonical_pass,
        canonical_pass_rate=canonical_pass / total if total > 0 else 1.0,
        required_pass_count=required_pass,
        required_pass_rate=required_pass / total if total > 0 else 1.0,
        failure_records=failure_records,
    )


# ── Fragmentation ───────────────────────────────────────────────────


@dataclass
class FragmentationMetrics:
    item_count: int
    total_tokens: int
    avg_tokens_per_item: float
    pct_one_token: float
    pct_over_four_tokens: float
    max_tokens: int


def _compute_fragmentation(
    records: list[EvaluationRecord],
    _metrics: list[RecordMetrics],
    tokenizer: BharatTokenizer,
) -> dict[str, FragmentationMetrics]:
    result: dict[str, FragmentationMetrics] = {}

    categories: dict[str, re.Pattern[str]] = dict(_FRAGMENTATION_PATTERNS)

    for name, pattern in categories.items():
        all_counts: list[int] = []
        for rec in records:
            text = rec.text
            items = pattern.findall(text)
            for item in items[:1000]:
                encoded = tokenizer.encode(item, add_special_tokens=False)
                all_counts.append(len(encoded))

        if not all_counts:
            result[name] = FragmentationMetrics(
                item_count=0,
                total_tokens=0,
                avg_tokens_per_item=0.0,
                pct_one_token=0.0,
                pct_over_four_tokens=0.0,
                max_tokens=0,
            )
            continue

        total_tokens = sum(all_counts)
        item_count = len(all_counts)
        avg = total_tokens / item_count if item_count > 0 else 0.0
        one_tok = sum(1 for c in all_counts if c <= 1)
        over_four = sum(1 for c in all_counts if c >= 4)
        max_tok = max(all_counts) if all_counts else 0

        result[name] = FragmentationMetrics(
            item_count=item_count,
            total_tokens=total_tokens,
            avg_tokens_per_item=avg,
            pct_one_token=one_tok / item_count * 100 if item_count > 0 else 0.0,
            pct_over_four_tokens=over_four / item_count * 100 if item_count > 0 else 0.0,
            max_tokens=max_tok,
        )

    return result


# ── Tokenizer protocol ──────────────────────────────────────────────


class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def fingerprint(self) -> str: ...


# ── Main evaluation class ────────────────────────────────────────────


class TokenizerEvaluation:
    def __init__(self, tokenizers: dict[str, BharatTokenizer]) -> None:
        if not tokenizers:
            msg = "at least one tokenizer required"
            raise ValueError(msg)
        self._tokenizers: dict[str, BharatTokenizer] = dict(tokenizers)
        self._records: list[EvaluationRecord] | None = None
        self._metrics: dict[str, list[RecordMetrics]] = {}

    def load_records(self, path: Path) -> None:
        self._records = _validate_jsonl(path)

    def set_records(self, records: list[EvaluationRecord]) -> None:
        for r in records:
            if not isinstance(r, EvaluationRecord):
                msg = "all records must be EvaluationRecord instances"
                raise ValueError(msg)
        self._records = list(records)

    def compute(self) -> dict[str, Any]:
        if self._records is None:
            msg = "no records loaded"
            raise ValueError(msg)

        self._metrics = {}
        for name, tokenizer in self._tokenizers.items():
            self._metrics[name] = [_compute_record_metrics(r, tokenizer) for r in self._records]

        return self._build_report()

    def get_detailed_records(self) -> list[dict[str, Any]]:
        records_out: list[dict[str, Any]] = []
        for name in sorted(self._tokenizers):
            metrics_list = self._metrics.get(name, [])
            for m in metrics_list:
                records_out.append(
                    {
                        "tokenizer": name,
                        "record_id": m.record_id,
                        "language": m.language,
                        "script": m.script,
                        "domain": m.domain,
                        "category": m.category,
                        "char_count": m.char_count,
                        "token_count": m.token_count,
                        "tokens_per_char": round(m.tokens_per_char, 6),
                        "unknown_token_count": m.unknown_token_count,
                        "special_token_count": m.special_token_count,
                        "byte_token_count": m.byte_token_count,
                        "merged_token_count": m.merged_token_count,
                        "exact_round_trip": m.exact_round_trip,
                        "nfc_round_trip": m.nfc_round_trip,
                        "canonical_round_trip": m.canonical_round_trip,
                        "required_pass": m.required_pass,
                    }
                )
        return records_out

    def _build_report(self) -> dict[str, Any]:
        if self._records is None:
            msg = "no records loaded"
            raise ValueError(msg)

        tokenizer_names = sorted(self._tokenizers.keys())
        tokenizer_fingerprints = {n: t.fingerprint() for n, t in self._tokenizers.items()}

        aggregate: dict[str, Any] = {}
        per_language: dict[str, dict[str, Any]] = {}
        per_script: dict[str, dict[str, Any]] = {}
        per_domain: dict[str, dict[str, Any]] = {}
        per_category: dict[str, dict[str, Any]] = {}
        round_trip_results: dict[str, Any] = {}
        fragmentation_results: dict[str, Any] = {}
        byte_coverage_results: dict[str, Any] = {}
        comparison_results: list[dict[str, Any]] = []
        failed_records: list[dict[str, str]] = []

        for name in tokenizer_names:
            metrics = self._metrics[name]
            agg = _compute_aggregate(metrics)
            aggregate[name] = {
                "record_count": agg.record_count,
                "char_count": agg.char_count,
                "byte_count": agg.byte_count,
                "token_count": agg.token_count,
                "micro_fertility": agg.micro_fertility,
                "macro_fertility": agg.macro_fertility,
                "min_fertility": agg.min_fertility,
                "max_fertility": agg.max_fertility,
                "median_fertility": agg.median_fertility,
            }

            by_lang: dict[str, list[RecordMetrics]] = {}
            for m in metrics:
                by_lang.setdefault(m.language, []).append(m)
            per_language[name] = {
                lang: _metrics_to_dict(_compute_aggregate(group))
                for lang, group in sorted(by_lang.items())
            }

            by_script: dict[str, list[RecordMetrics]] = {}
            for m in metrics:
                by_script.setdefault(m.script, []).append(m)
            per_script[name] = {
                script: _metrics_to_dict(_compute_aggregate(group))
                for script, group in sorted(by_script.items())
            }

            by_domain: dict[str, list[RecordMetrics]] = {}
            for m in metrics:
                by_domain.setdefault(m.domain, []).append(m)
            per_domain[name] = {
                dom: _metrics_to_dict(_compute_aggregate(group))
                for dom, group in sorted(by_domain.items())
            }

            by_category: dict[str, list[RecordMetrics]] = {}
            for m in metrics:
                by_category.setdefault(m.category, []).append(m)
            per_category[name] = {
                cat: _metrics_to_dict(_compute_aggregate(group))
                for cat, group in sorted(by_category.items())
            }

            rt = _compute_round_trip(metrics, self._records)
            round_trip_results[name] = {
                "exact_pass_count": rt.exact_pass_count,
                "exact_pass_rate": rt.exact_pass_rate,
                "nfc_pass_count": rt.nfc_pass_count,
                "nfc_pass_rate": rt.nfc_pass_rate,
                "canonical_pass_count": rt.canonical_pass_count,
                "canonical_pass_rate": rt.canonical_pass_rate,
                "required_pass_count": rt.required_pass_count,
                "required_pass_rate": rt.required_pass_rate,
                "failure_records": rt.failure_records,
            }

            for fr in rt.failure_records:
                failed_records.append(
                    {
                        "record_id": fr["record_id"],
                        "tokenizer": name,
                        "failure_reasons": fr["failure_reasons"],
                    }
                )

            frag = _compute_fragmentation(self._records, metrics, self._tokenizers[name])
            fragmentation_results[name] = {
                cat: {
                    "item_count": f.item_count,
                    "total_tokens": f.total_tokens,
                    "avg_tokens_per_item": round(f.avg_tokens_per_item, 4),
                    "pct_one_token": round(f.pct_one_token, 2),
                    "pct_over_four_tokens": round(f.pct_over_four_tokens, 2),
                    "max_tokens": f.max_tokens,
                }
                for cat, f in sorted(frag.items())
            }

            bc = check_byte_alphabet(self._tokenizers[name])
            byte_coverage_results[name] = bc

            unknown_counts = sum(m.unknown_token_count for m in metrics)
            records_with_unknown = sum(1 for m in metrics if m.unknown_token_count > 0)
            aggregate[name]["unknown_token_count"] = unknown_counts
            aggregate[name]["records_with_unknown"] = records_with_unknown
            aggregate[name]["unknown_token_rate"] = (
                unknown_counts / aggregate[name]["token_count"]
                if aggregate[name]["token_count"] > 0
                else 0.0
            )
            aggregate[name]["special_token_count"] = sum(m.special_token_count for m in metrics)
            byte_token_total = sum(
                m.byte_token_count for m in metrics if m.byte_token_count is not None
            )
            merged_token_total = sum(
                m.merged_token_count for m in metrics if m.merged_token_count is not None
            )
            aggregate[name]["byte_token_count"] = byte_token_total
            aggregate[name]["merged_token_count"] = merged_token_total

        if len(tokenizer_names) >= 2:
            for i in range(len(tokenizer_names)):
                for j in range(i + 1, len(tokenizer_names)):
                    a_name = tokenizer_names[i]
                    b_name = tokenizer_names[j]
                    a_agg = aggregate[a_name]
                    b_agg = aggregate[b_name]

                    abs_diff = abs(a_agg["token_count"] - b_agg["token_count"])
                    rel_fert_diff = (
                        (a_agg["micro_fertility"] - b_agg["micro_fertility"])
                        / b_agg["micro_fertility"]
                        if b_agg["micro_fertility"] > 0
                        else 0.0
                    )

                    per_lang_fert_diff: dict[str, float] = {}
                    all_langs = set(per_language.get(a_name, {})) | set(
                        per_language.get(b_name, {})
                    )
                    for lang in sorted(all_langs):
                        a_lang = per_language.get(a_name, {}).get(lang, {})
                        b_lang = per_language.get(b_name, {}).get(lang, {})
                        a_fert = a_lang.get("micro_fertility", 0.0)
                        b_fert = b_lang.get("micro_fertility", 0.0)
                        per_lang_fert_diff[lang] = a_fert - b_fert

                    metrics_a = self._metrics[a_name]
                    metrics_b = self._metrics[b_name]
                    a_by_id = {m.record_id: m for m in metrics_a}
                    b_by_id = {m.record_id: m for m in metrics_b}
                    eligible = 0
                    excluded = 0
                    excluded_ids: list[dict[str, str]] = []
                    wins_a = wins_b = ties = 0
                    all_ids = sorted(set(a_by_id) | set(b_by_id))
                    for rid in all_ids:
                        m_a = a_by_id.get(rid)
                        m_b = b_by_id.get(rid)
                        if m_a is None or m_b is None:
                            excluded += 1
                            excluded_ids.append(
                                {"record_id": rid, "reason": "missing_in_one_tokenizer"}
                            )
                            continue
                        if not m_a.required_pass or not m_b.required_pass:
                            excluded += 1
                            excluded_ids.append({"record_id": rid, "reason": "round_trip_failure"})
                            continue
                        eligible += 1
                        if m_a.token_count < m_b.token_count:
                            wins_a += 1
                        elif m_b.token_count < m_a.token_count:
                            wins_b += 1
                        else:
                            ties += 1

                    rt_diff: dict[str, int] = {}
                    rt_a = round_trip_results.get(a_name, {})
                    rt_b = round_trip_results.get(b_name, {})
                    rt_diff["required_pass_a"] = rt_a.get("required_pass_count", 0)
                    rt_diff["required_pass_b"] = rt_b.get("required_pass_count", 0)

                    unk_diff: dict[str, int] = {}
                    unk_diff["unknown_a"] = a_agg.get("unknown_token_count", 0)
                    unk_diff["unknown_b"] = b_agg.get("unknown_token_count", 0)

                    comparison_results.append(
                        {
                            "tokenizer_a": a_name,
                            "tokenizer_b": b_name,
                            "absolute_token_count_difference": abs_diff,
                            "relative_fertility_difference": round(rel_fert_diff, 6),
                            "per_language_fertility_difference": per_lang_fert_diff,
                            "eligible_record_count": eligible,
                            "excluded_record_count": excluded,
                            "excluded_records": excluded_ids,
                            "wins_a": wins_a,
                            "wins_b": wins_b,
                            "ties": ties,
                            "round_trip_difference": rt_diff,
                            "unknown_token_difference": unk_diff,
                        }
                    )

        failed_records.sort(key=lambda x: (x["tokenizer"], x["record_id"]))

        dataset_hash = self._compute_dataset_hash()

        report_payload = {
            "schema_version": _SCHEMA_VERSION,
            "evaluator_version": _EVALUATOR_VERSION,
            "input_dataset_sha256": dataset_hash,
            "tokenizer_names": tokenizer_names,
            "tokenizer_fingerprints": tokenizer_fingerprints,
            "aggregate": aggregate,
            "per_language": per_language,
            "per_script": per_script,
            "per_domain": per_domain,
            "per_category": per_category,
            "round_trip": round_trip_results,
            "fragmentation": fragmentation_results,
            "byte_coverage": byte_coverage_results,
            "comparison": comparison_results,
            "failed_records": failed_records,
        }

        report_payload["report_sha256"] = self._compute_report_digest(report_payload)
        return report_payload

    def _compute_dataset_hash(self) -> str:
        if self._records is None:
            return ""
        ordered = sorted(self._records, key=lambda r: r.record_id)
        h = hashlib.sha256()
        for r in ordered:
            obj = {
                "record_id": r.record_id,
                "language": r.language,
                "script": r.script,
                "domain": r.domain,
                "text": r.text,
                "tags": sorted(r.tags),
                "canonical_equivalent": r.canonical_equivalent,
                "category": r.category,
            }
            canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            h.update(canonical.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _compute_report_digest(payload: dict[str, Any]) -> str:
        excluded = payload.copy()
        excluded.pop("report_sha256", None)
        canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def serialize_report(report: dict[str, Any]) -> str:
        return json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _metrics_to_dict(agg: AggregateMetrics) -> dict[str, Any]:
    return {
        "record_count": agg.record_count,
        "char_count": agg.char_count,
        "byte_count": agg.byte_count,
        "token_count": agg.token_count,
        "micro_fertility": round(agg.micro_fertility, 6),
        "macro_fertility": round(agg.macro_fertility, 6),
        "min_fertility": round(agg.min_fertility, 6),
        "max_fertility": round(agg.max_fertility, 6),
        "median_fertility": round(agg.median_fertility, 6),
    }
