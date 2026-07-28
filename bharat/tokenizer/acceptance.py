from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_SCHEMA_VERSION = "tokenizer-acceptance-v1"


@dataclass(frozen=True)
class TokenizerAcceptanceThresholds:
    min_record_count: int = 1
    min_required_round_trip_rate: float = 1.0
    max_unknown_token_rate: float = 0.0
    require_complete_byte_coverage: bool = True
    max_micro_fertility: float | None = None
    max_language_micro_fertility: float | None = None

    def __post_init__(self) -> None:
        if self.min_record_count < 1:
            raise ValueError("min_record_count must be at least 1")
        for name, value in (
            ("min_required_round_trip_rate", self.min_required_round_trip_rate),
            ("max_unknown_token_rate", self.max_unknown_token_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (
            ("max_micro_fertility", self.max_micro_fertility),
            ("max_language_micro_fertility", self.max_language_micro_fertility),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when set")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenizerAcceptanceThresholds:
        allowed = {
            "min_record_count",
            "min_required_round_trip_rate",
            "max_unknown_token_rate",
            "require_complete_byte_coverage",
            "max_micro_fertility",
            "max_language_micro_fertility",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown threshold fields: {', '.join(unknown)}")
        return cls(**payload)


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


def evaluate_tokenizer_acceptance(
    report: dict[str, Any],
    tokenizer_name: str,
    thresholds: TokenizerAcceptanceThresholds,
) -> dict[str, Any]:
    names = report.get("tokenizer_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("report tokenizer_names must be a list of strings")
    if tokenizer_name not in names:
        raise ValueError(f"tokenizer {tokenizer_name!r} is not present in report")

    aggregate = _require_mapping(report, "aggregate")
    round_trip = _require_mapping(report, "round_trip")
    byte_coverage = _require_mapping(report, "byte_coverage")
    per_language = _require_mapping(report, "per_language")

    tokenizer_aggregate = _require_named_mapping(aggregate, tokenizer_name, "aggregate")
    tokenizer_round_trip = _require_named_mapping(
        round_trip, tokenizer_name, "round_trip"
    )
    tokenizer_byte_coverage = _require_named_mapping(
        byte_coverage, tokenizer_name, "byte_coverage"
    )
    tokenizer_languages = _require_named_mapping(
        per_language, tokenizer_name, "per_language"
    )

    record_count = _require_int(tokenizer_aggregate, "record_count")
    required_rate = _require_number(tokenizer_round_trip, "required_pass_rate")
    unknown_rate = _require_number(tokenizer_aggregate, "unknown_token_rate")
    micro_fertility = _require_number(tokenizer_aggregate, "micro_fertility")
    byte_complete = tokenizer_byte_coverage.get("complete") is True

    checks = [
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

    if thresholds.require_complete_byte_coverage:
        checks.append(
            AcceptanceCheck(
                "complete_byte_coverage",
                byte_complete,
                byte_complete,
                "true",
            )
        )

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

    checks_payload = [check.to_dict() for check in checks]
    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "tokenizer_name": tokenizer_name,
        "input_report_sha256": report.get("report_sha256"),
        "passed": all(check.passed for check in checks),
        "checks": checks_payload,
    }
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    result["acceptance_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return result


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
