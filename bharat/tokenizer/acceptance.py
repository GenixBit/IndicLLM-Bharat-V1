from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from bharat.tokenizer.evaluation import validate_evaluation_report

_SCHEMA_VERSION = "tokenizer-acceptance-v1"
_THRESHOLD_SCHEMA_VERSION = "tokenizer-acceptance-thresholds-v1"
_SUPPORTED_EVALUATOR_VERSIONS: tuple[str, ...] = ("1.0.3",)
_ALLOWED_CONFIG_FIELDS: set[str] = {
    "schema_version",
    "status",
    "evidence_scope",
    "notes",
    "thresholds",
}
_ALLOWED_STATUS_VALUES: set[str] = {"provisional", "production"}
_ALLOWED_EVIDENCE_SCOPE_VALUES: set[str] = {"synthetic-local-only", "approved-evaluation-set"}


# ── Threshold configuration ─────────────────────────────────────────


@dataclass(frozen=True)
class TokenizerAcceptanceThresholds:
    min_record_count: int = 1
    min_required_round_trip_rate: float = 1.0
    min_canonical_pass_rate: float = 1.0
    max_unknown_token_rate: float = 0.0
    require_complete_byte_coverage: bool = True
    max_micro_fertility: float | None = None
    max_language_micro_fertility: float | None = None
    min_canonical_evaluated_count: int = 0
    required_languages: tuple[str, ...] = ()
    min_records_per_required_language: int = 1

    _INT_FIELDS = (
        "min_record_count",
        "min_records_per_required_language",
        "min_canonical_evaluated_count",
    )
    _RATE_FIELDS = (
        "min_required_round_trip_rate",
        "min_canonical_pass_rate",
        "max_unknown_token_rate",
    )
    _OPTIONAL_FLOAT_FIELDS = ("max_micro_fertility", "max_language_micro_fertility")
    _BOOL_FIELDS = ("require_complete_byte_coverage",)
    _TUPLE_FIELDS = ("required_languages",)

    def __post_init__(self) -> None:
        if self.min_record_count < 1:
            raise ValueError("min_record_count must be at least 1")
        for name, value in (
            ("min_required_round_trip_rate", self.min_required_round_trip_rate),
            ("min_canonical_pass_rate", self.min_canonical_pass_rate),
            ("max_unknown_token_rate", self.max_unknown_token_rate),
        ):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        for name, value in (
            ("max_micro_fertility", self.max_micro_fertility),
            ("max_language_micro_fertility", self.max_language_micro_fertility),
        ):
            if value is not None:
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ValueError(f"{name} must be a finite number")
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if value <= 0.0:
                    raise ValueError(f"{name} must be positive when set")

        if self.min_records_per_required_language < 1:
            raise ValueError("min_records_per_required_language must be at least 1")
        if self.min_canonical_evaluated_count < 0:
            raise ValueError("min_canonical_evaluated_count must be non-negative")

    def to_canonical_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                result[f.name] = list(v)
            else:
                result[f.name] = v
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TokenizerAcceptanceThresholds:
        allowed = {
            "min_record_count",
            "min_required_round_trip_rate",
            "min_canonical_pass_rate",
            "min_canonical_evaluated_count",
            "max_unknown_token_rate",
            "require_complete_byte_coverage",
            "max_micro_fertility",
            "max_language_micro_fertility",
            "required_languages",
            "min_records_per_required_language",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown threshold fields: {', '.join(unknown)}")

        for name in cls._INT_FIELDS:
            v = payload.get(name)
            if v is not None:
                if not (isinstance(v, int) and not isinstance(v, bool)):
                    raise ValueError(
                        f"threshold {name!r} must be an integer, got {type(v).__name__}"
                    )
                if name == "min_canonical_evaluated_count":
                    if v < 0:
                        raise ValueError(f"threshold {name!r} must be >= 0")
                elif v < 1:
                    raise ValueError(f"threshold {name!r} must be >= 1")

        for name in cls._RATE_FIELDS:
            v = payload.get(name)
            if v is not None:
                if not isinstance(v, int | float) or isinstance(v, bool):
                    raise ValueError(f"threshold {name!r} must be a number, got {type(v).__name__}")
                if not math.isfinite(v):
                    raise ValueError(f"threshold {name!r} must be a finite number")
                if not 0.0 <= float(v) <= 1.0:
                    raise ValueError(f"threshold {name!r} must be between 0 and 1")

        for name in cls._OPTIONAL_FLOAT_FIELDS:
            v = payload.get(name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int | float)):
                raise ValueError(
                    f"threshold {name!r} must be a number or null, " f"got {type(v).__name__}"
                )
            if v is not None and not math.isfinite(v):
                raise ValueError(f"threshold {name!r} must be a finite number")

        for name in cls._BOOL_FIELDS:
            v = payload.get(name)
            if v is not None and not isinstance(v, bool):
                raise ValueError(f"threshold {name!r} must be a boolean, got {type(v).__name__}")

        for name in cls._TUPLE_FIELDS:
            v = payload.get(name)
            if v is not None:
                if not isinstance(v, list | tuple):
                    raise ValueError(f"threshold {name!r} must be a list, got {type(v).__name__}")
                seen: set[str] = set()
                for item in v:
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(f"threshold {name!r} items must be non-empty strings")
                    if item.strip() in seen:
                        raise ValueError(f"threshold {name!r} contains duplicate: {item.strip()!r}")
                    seen.add(item.strip())

        filtered = {k: v for k, v in payload.items() if k in allowed and v is not None}
        return cls(**filtered)


# ── Full threshold configuration model ──────────────────────────────


@dataclass(frozen=True)
class ThresholdConfiguration:
    schema_version: str
    thresholds: TokenizerAcceptanceThresholds
    status: str = "provisional"
    evidence_scope: str = "synthetic-local-only"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != _THRESHOLD_SCHEMA_VERSION:
            msg = f"unsupported schema_version: {self.schema_version!r}"
            raise ValueError(msg)
        if self.status not in _ALLOWED_STATUS_VALUES:
            msg = f"unknown status: {self.status!r}"
            raise ValueError(msg)
        if self.evidence_scope not in _ALLOWED_EVIDENCE_SCOPE_VALUES:
            msg = f"unknown evidence_scope: {self.evidence_scope!r}"
            raise ValueError(msg)
        if self.status == "production" and self.evidence_scope != "approved-evaluation-set":
            msg = (
                f"status=production requires evidence_scope="
                f"'approved-evaluation-set', got {self.evidence_scope!r}"
            )
            raise ValueError(msg)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "evidence_scope": self.evidence_scope,
            "notes": list(self.notes),
            "thresholds": self.thresholds.to_canonical_dict(),
        }

    def configuration_sha256(self) -> str:
        canonical = json.dumps(
            self.to_canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ThresholdConfiguration:
        unknown = sorted(set(payload) - _ALLOWED_CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")

        sv = payload.get("schema_version")
        if sv != _THRESHOLD_SCHEMA_VERSION:
            msg = f"unsupported schema_version: {sv!r}"
            raise ValueError(msg)

        status = payload.get("status", "provisional")
        if status not in _ALLOWED_STATUS_VALUES:
            raise ValueError(f"unknown status: {status!r}")

        evidence_scope = payload.get("evidence_scope", "synthetic-local-only")
        if evidence_scope not in _ALLOWED_EVIDENCE_SCOPE_VALUES:
            raise ValueError(f"unknown evidence_scope: {evidence_scope!r}")

        notes_raw = payload.get("notes", [])
        if not isinstance(notes_raw, list):
            raise ValueError("notes must be a list")
        if not all(isinstance(n, str) for n in notes_raw):
            raise ValueError("all notes must be strings")
        notes = tuple(notes_raw)

        raw_thresholds = payload.get("thresholds")
        if not isinstance(raw_thresholds, dict):
            raise ValueError("thresholds must be an object")
        thresholds = TokenizerAcceptanceThresholds.from_dict(raw_thresholds)

        return cls(
            schema_version=sv,
            status=status,
            evidence_scope=evidence_scope,
            notes=notes,
            thresholds=thresholds,
        )


# ── Acceptance check ────────────────────────────────────────────────


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    actual: Any
    expected: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual": self.actual,
            "expected": self.expected,
            "name": self.name,
            "passed": self.passed,
        }


# ── Core evaluation gate ────────────────────────────────────────────


def evaluate_tokenizer_acceptance(
    report: dict[str, Any],
    tokenizer_name: str,
    threshold_config: ThresholdConfiguration,
) -> dict[str, Any]:
    """Evaluate whether *report* satisfies *threshold_config* for *tokenizer_name*.

    Steps:
    1. Validate the evaluation report integrity and structure.
    2. Validate report schema and version.
    3. Extract and validate the tokenizer-specific sections.
    4. Run all configured checks.
    5. Build a cryptographically-bound acceptance result.
    """
    validate_evaluation_report(report)

    names = report["tokenizer_names"]
    if tokenizer_name not in names:
        msg = f"tokenizer {tokenizer_name!r} is not present in report"
        raise ValueError(msg)

    if report.get("evaluator_version") not in _SUPPORTED_EVALUATOR_VERSIONS:
        msg = f"unsupported evaluator_version: " f"{report.get('evaluator_version')!r}"
        raise ValueError(msg)

    aggregate = _require_mapping(report, "aggregate")
    round_trip = _require_mapping(report, "round_trip")
    byte_coverage = _require_mapping(report, "byte_coverage")
    per_language = _require_mapping(report, "per_language")

    tokenizer_aggregate = _require_named_mapping(aggregate, tokenizer_name, "aggregate")
    tokenizer_round_trip = _require_named_mapping(round_trip, tokenizer_name, "round_trip")
    tokenizer_byte_coverage = _require_named_mapping(byte_coverage, tokenizer_name, "byte_coverage")
    tokenizer_languages = _require_named_mapping(per_language, tokenizer_name, "per_language")

    fingerprint = report.get("tokenizer_fingerprints", {}).get(tokenizer_name, "")
    input_dataset_sha256 = report.get("input_dataset_sha256", "")
    input_report_sha256 = report.get("report_sha256", "")

    thresholds = threshold_config.thresholds

    record_count = _require_int(tokenizer_aggregate, "record_count")
    required_rate = _require_number(tokenizer_round_trip, "required_pass_rate")
    unknown_rate = _require_number(tokenizer_aggregate, "unknown_token_rate")
    micro_fertility = _require_number(tokenizer_aggregate, "micro_fertility")

    for val, label in [
        (required_rate, "required_pass_rate"),
        (unknown_rate, "unknown_token_rate"),
        (micro_fertility, "micro_fertility"),
    ]:
        if not math.isfinite(val):
            msg = f"report metric {label!r} must be finite"
            raise ValueError(msg)

    checks: list[AcceptanceCheck] = [
        AcceptanceCheck(
            "record_count",
            record_count >= thresholds.min_record_count,
            record_count,
            f">= {thresholds.min_record_count}",
        ),
        AcceptanceCheck(
            "required_round_trip_rate",
            required_rate >= thresholds.min_required_round_trip_rate,
            required_rate,
            f">= {thresholds.min_required_round_trip_rate}",
        ),
        AcceptanceCheck(
            "unknown_token_rate",
            unknown_rate <= thresholds.max_unknown_token_rate,
            unknown_rate,
            f"<= {thresholds.max_unknown_token_rate}",
        ),
    ]

    _add_byte_coverage_checks(checks, tokenizer_byte_coverage, thresholds)
    _add_fertility_checks(checks, micro_fertility, tokenizer_languages, thresholds)
    _add_canonical_pass_checks(checks, tokenizer_round_trip, thresholds)
    _add_language_presence_checks(checks, tokenizer_languages, thresholds)

    checks_payload = [check.to_dict() for check in checks]
    overall_passed = all(check.passed for check in checks)

    thresholds_canonical = thresholds.to_canonical_dict()
    thresholds_sha256 = hashlib.sha256(
        json.dumps(
            thresholds_canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    config_sha256 = threshold_config.configuration_sha256()

    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_schema_version": report.get("schema_version", ""),
        "evaluator_version": report.get("evaluator_version", ""),
        "tokenizer_name": tokenizer_name,
        "tokenizer_fingerprint": fingerprint,
        "input_dataset_sha256": input_dataset_sha256,
        "input_report_sha256": input_report_sha256,
        "threshold_schema_version": _THRESHOLD_SCHEMA_VERSION,
        "threshold_configuration_status": threshold_config.status,
        "threshold_evidence_scope": threshold_config.evidence_scope,
        "threshold_configuration_sha256": config_sha256,
        "thresholds": thresholds_canonical,
        "thresholds_sha256": thresholds_sha256,
        "checks": checks_payload,
        "passed": overall_passed,
    }

    result_canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    result["acceptance_sha256"] = hashlib.sha256(result_canonical.encode("utf-8")).hexdigest()

    return result


def _add_byte_coverage_checks(
    checks: list[AcceptanceCheck],
    byte_coverage: dict[str, Any],
    thresholds: TokenizerAcceptanceThresholds,
) -> None:
    if not thresholds.require_complete_byte_coverage:
        return

    status = byte_coverage.get("status")
    complete = byte_coverage.get("complete")
    reachable = byte_coverage.get("reachable_count")
    missing = byte_coverage.get("missing_byte_values")

    if status == "complete" and complete is True and reachable == 256:
        passes = isinstance(missing, list) and len(missing) == 0
    else:
        passes = False

    checks.append(
        AcceptanceCheck(
            "complete_byte_coverage",
            passes,
            {
                "status": status,
                "complete": complete,
                "reachable_count": reachable,
                "missing_byte_values_len": len(missing) if isinstance(missing, list) else -1,
            },
            "status=complete, complete=True, reachable_count=256, missing=[]",
        )
    )


def _add_fertility_checks(
    checks: list[AcceptanceCheck],
    micro_fertility: float,
    tokenizer_languages: dict[str, Any],
    thresholds: TokenizerAcceptanceThresholds,
) -> None:
    if thresholds.max_micro_fertility is not None:
        checks.append(
            AcceptanceCheck(
                "micro_fertility",
                micro_fertility <= thresholds.max_micro_fertility,
                micro_fertility,
                f"<= {thresholds.max_micro_fertility}",
            )
        )

    if thresholds.max_language_micro_fertility is not None:
        language_failures: dict[str, float] = {}
        for language, metrics in sorted(tokenizer_languages.items()):
            if not isinstance(language, str) or not isinstance(metrics, dict):
                raise ValueError("per_language entries must map strings to objects")
            fertility = _require_number(metrics, "micro_fertility")
            if not math.isfinite(fertility):
                msg = f"per_language.{language}.micro_fertility must be finite"
                raise ValueError(msg)
            if fertility > thresholds.max_language_micro_fertility:
                language_failures[language] = fertility
        checks.append(
            AcceptanceCheck(
                "language_micro_fertility",
                not language_failures,
                language_failures,
                f"all <= {thresholds.max_language_micro_fertility}",
            )
        )


def _add_canonical_pass_checks(
    checks: list[AcceptanceCheck],
    round_trip: dict[str, Any],
    thresholds: TokenizerAcceptanceThresholds,
) -> None:
    canonical_pass_rate = round_trip.get("canonical_pass_rate")
    if canonical_pass_rate is not None:
        if (
            not isinstance(canonical_pass_rate, int | float)
            or isinstance(canonical_pass_rate, bool)
            or not math.isfinite(canonical_pass_rate)
        ):
            passed = False
        else:
            passed = float(canonical_pass_rate) >= thresholds.min_canonical_pass_rate

        checks.append(
            AcceptanceCheck(
                "canonical_equivalent_pass_rate",
                passed,
                canonical_pass_rate,
                f">= {thresholds.min_canonical_pass_rate}",
            )
        )

    if thresholds.min_canonical_evaluated_count > 0:
        evaluated_count = round_trip.get("canonical_evaluated_count", 0)
        if not isinstance(evaluated_count, int) or isinstance(evaluated_count, bool):
            evaluated_count = 0
        checks.append(
            AcceptanceCheck(
                "canonical_evaluated_count",
                evaluated_count >= thresholds.min_canonical_evaluated_count,
                evaluated_count,
                f">= {thresholds.min_canonical_evaluated_count}",
            )
        )


def _add_language_presence_checks(
    checks: list[AcceptanceCheck],
    tokenizer_languages: dict[str, Any],
    thresholds: TokenizerAcceptanceThresholds,
) -> None:
    if not thresholds.required_languages:
        return

    missing_langs: list[str] = []
    insufficient_langs: dict[str, int] = {}
    for lang in thresholds.required_languages:
        if lang not in tokenizer_languages:
            missing_langs.append(lang)
        else:
            lang_metrics = tokenizer_languages[lang]
            if not isinstance(lang_metrics, dict):
                raise ValueError(f"per_language.{lang} must be an object")
            rc = lang_metrics.get("record_count", 0)
            if not (isinstance(rc, int) and not isinstance(rc, bool)):
                rc = 0
            if rc < thresholds.min_records_per_required_language:
                insufficient_langs[lang] = rc

    passed = len(missing_langs) == 0 and len(insufficient_langs) == 0
    details: dict[str, Any] = {}
    if missing_langs:
        details["missing_languages"] = missing_langs
    if insufficient_langs:
        details["insufficient_records"] = {k: v for k, v in sorted(insufficient_langs.items())}
    checks.append(
        AcceptanceCheck(
            "required_languages_present",
            passed,
            details if details else "all present",
            (
                f"languages={list(thresholds.required_languages)}, "
                f"min_records={thresholds.min_records_per_required_language}"
            ),
        )
    )


# ── Internal helpers ────────────────────────────────────────────────


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"report field {key!r} must be an object")
    return value


def _require_named_mapping(
    payload: dict[str, Any],
    name: str,
    field_name: str,
) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"report field {field_name!r} has no object for {name!r}")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"report metric {key!r} must be an integer")
    return value


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"report metric {key!r} must be numeric")
    return float(value)
