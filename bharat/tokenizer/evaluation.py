from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Protocol

from bharat.tokenizer.base import BharatTokenizer

_SCHEMA_VERSION = "eval-v1"
_EVALUATOR_VERSION = "1.0.0"

_RE_WHITESPACE = re.compile(r"\s+")
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
        for surrogate in re.findall(r"[\ud800-\udfff]", self.text):
            msg = f"lone surrogate U+{ord(surrogate):04X} in record {self.record_id}"
            raise ValueError(msg)


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
        for surrogate in re.findall(r"[\ud800-\udfff]", text):
            msg = f"line {line_num}: lone surrogate U+{ord(surrogate):04X}"
            raise ValueError(msg)

        tags_raw = obj.get("tags")
        if tags_raw is not None and (
            not isinstance(tags_raw, list) or not all(isinstance(t, str) for t in tags_raw)
        ):
            msg = f"line {line_num}: tags must be a list of strings"
            raise ValueError(msg)
        tags = tuple(tags_raw) if tags_raw else ()

        canonical = obj.get("canonical_equivalent")
        if canonical is not None and not isinstance(canonical, str):
            msg = f"line {line_num}: canonical_equivalent must be a string"
            raise ValueError(msg)

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
    canonical_equivalent: bool = True


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
    canonical_equivalent = True
    if record.canonical_equivalent is not None:
        canonical_ids = tokenizer.encode(record.canonical_equivalent, add_special_tokens=False)
        canonical_decoded = tokenizer.decode(canonical_ids)
        canonical_equivalent = canonical_decoded == nfc_text

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
        canonical_equivalent=canonical_equivalent,
    )


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


@dataclass
class RoundTripSummary:
    exact_pass_count: int
    exact_pass_rate: float
    nfc_pass_count: int
    nfc_pass_rate: float
    failed_record_ids: list[str]
    failure_categories: dict[str, int]


def _compute_round_trip(metrics: list[RecordMetrics]) -> RoundTripSummary:
    exact_pass = sum(1 for m in metrics if m.exact_round_trip)
    nfc_pass = sum(1 for m in metrics if m.nfc_round_trip)
    total = len(metrics)

    failed = [m.record_id for m in metrics if not m.exact_round_trip or not m.nfc_round_trip]
    categories: dict[str, int] = Counter(
        "exact" if not m.exact_round_trip else "nfc"
        for m in metrics
        if not m.exact_round_trip or not m.nfc_round_trip
    )

    return RoundTripSummary(
        exact_pass_count=exact_pass,
        exact_pass_rate=exact_pass / total if total > 0 else 1.0,
        nfc_pass_count=nfc_pass,
        nfc_pass_rate=nfc_pass / total if total > 0 else 1.0,
        failed_record_ids=sorted(failed),
        failure_categories=dict(categories),
    )


@dataclass
class FragmentationMetrics:
    item_count: int
    total_tokens: int
    avg_tokens_per_item: float
    pct_one_token: float
    pct_over_four_tokens: float
    max_tokens: int


def _compute_fragmentation(metrics: list[RecordMetrics]) -> dict[str, FragmentationMetrics]:
    result: dict[str, FragmentationMetrics] = {}
    all_texts = " ".join(m.decoded_text for m in metrics)

    categories: dict[str, re.Pattern[str]] = {
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

    for name, pattern in categories.items():
        items = pattern.findall(all_texts)
        if not items:
            result[name] = FragmentationMetrics(
                item_count=0,
                total_tokens=0,
                avg_tokens_per_item=0.0,
                pct_one_token=0.0,
                pct_over_four_tokens=0.0,
                max_tokens=0,
            )
            continue

        token_counts: list[int] = []
        for item in items[:1000]:
            tok_count = len(_RE_WHITESPACE.split(item))
            token_counts.append(max(tok_count, 1))

        total_tokens = sum(token_counts)
        item_count = len(token_counts)
        avg = total_tokens / item_count if item_count > 0 else 0.0
        one_tok = sum(1 for c in token_counts if c <= 1)
        over_four = sum(1 for c in token_counts if c >= 4)
        max_tok = max(token_counts) if token_counts else 0

        result[name] = FragmentationMetrics(
            item_count=item_count,
            total_tokens=total_tokens,
            avg_tokens_per_item=avg,
            pct_one_token=one_tok / item_count * 100 if item_count > 0 else 0.0,
            pct_over_four_tokens=over_four / item_count * 100 if item_count > 0 else 0.0,
            max_tokens=max_tok,
        )

    return result


@dataclass(frozen=True)
class ComparisonResult:
    tokenizer_a_name: str
    tokenizer_b_name: str
    abs_token_count_diff: int
    relative_fertility_diff: float
    per_language_fertility_diff: dict[str, float]
    wins_a: int
    wins_b: int
    ties: int
    round_trip_diff: dict[str, int]
    unknown_token_diff: dict[str, int]


class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def fingerprint(self) -> str: ...


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

    def _build_report(self) -> dict[str, Any]:
        if self._records is None:
            msg = "no records loaded"
            raise ValueError(msg)

        tokenizer_names = list(self._tokenizers.keys())
        tokenizer_fingerprints = {n: t.fingerprint() for n, t in self._tokenizers.items()}

        aggregate: dict[str, Any] = {}
        per_language: dict[str, dict[str, Any]] = {}
        per_script: dict[str, dict[str, Any]] = {}
        per_domain: dict[str, dict[str, Any]] = {}
        per_category: dict[str, dict[str, Any]] = {}
        round_trip_results: dict[str, Any] = {}
        fragmentation_results: dict[str, Any] = {}
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

            rt = _compute_round_trip(metrics)
            round_trip_results[name] = {
                "exact_pass_count": rt.exact_pass_count,
                "exact_pass_rate": rt.exact_pass_rate,
                "nfc_pass_count": rt.nfc_pass_count,
                "nfc_pass_rate": rt.nfc_pass_rate,
                "failed_record_ids": rt.failed_record_ids,
                "failure_categories": rt.failure_categories,
            }

            for fid in rt.failed_record_ids:
                failed_records.append({"record_id": fid, "tokenizer": name})

            frag = _compute_fragmentation(metrics)
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

                    wins_a = wins_b = ties = 0
                    for m_a, m_b in zip(self._metrics[a_name], self._metrics[b_name], strict=False):
                        if not m_a.exact_round_trip or not m_b.exact_round_trip:
                            continue
                        if m_a.token_count < m_b.token_count:
                            wins_a += 1
                        elif m_b.token_count < m_a.token_count:
                            wins_b += 1
                        else:
                            ties += 1

                    rt_diff: dict[str, int] = {}
                    rt_a = round_trip_results.get(a_name, {})
                    rt_b = round_trip_results.get(b_name, {})
                    rt_diff["exact_a_pass"] = rt_a.get("exact_pass_count", 0)
                    rt_diff["exact_b_pass"] = rt_b.get("exact_pass_count", 0)
                    rt_diff["nfc_a_pass"] = rt_a.get("nfc_pass_count", 0)
                    rt_diff["nfc_b_pass"] = rt_b.get("nfc_pass_count", 0)

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
            h.update(r.record_id.encode("utf-8"))
            h.update(r.text.encode("utf-8"))
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


def is_byte_alphabet_complete(tokenizer: BharatTokenizer) -> dict[str, Any]:
    complete = False
    missing: list[int] = []
    try:
        for b in range(256):
            try:
                ids = tokenizer.encode(bytes([b]).decode("latin-1"), add_special_tokens=False)
                if not ids:
                    missing.append(b)
            except (ValueError, UnicodeDecodeError):
                missing.append(b)
        complete = len(missing) == 0
    except Exception:
        complete = False
        missing = list(range(256))

    return {
        "complete": complete,
        "missing_byte_values": missing,
        "total_reachable": 256 - len(missing),
    }
