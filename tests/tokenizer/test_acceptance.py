from __future__ import annotations

import copy

import pytest

from bharat.tokenizer.acceptance import (
    TokenizerAcceptanceThresholds,
    evaluate_tokenizer_acceptance,
)


def _report() -> dict[str, object]:
    return {
        "report_sha256": "a" * 64,
        "tokenizer_names": ["bharat-bpe"],
        "aggregate": {
            "bharat-bpe": {
                "record_count": 12,
                "unknown_token_rate": 0.0,
                "micro_fertility": 1.25,
            }
        },
        "round_trip": {"bharat-bpe": {"required_pass_rate": 1.0}},
        "byte_coverage": {"bharat-bpe": {"complete": True}},
        "per_language": {
            "bharat-bpe": {
                "en": {"micro_fertility": 1.0},
                "hi": {"micro_fertility": 1.5},
            }
        },
    }


def _thresholds() -> TokenizerAcceptanceThresholds:
    return TokenizerAcceptanceThresholds(
        min_record_count=10,
        min_required_round_trip_rate=1.0,
        max_unknown_token_rate=0.0,
        require_complete_byte_coverage=True,
        max_micro_fertility=2.0,
        max_language_micro_fertility=2.0,
    )


def test_acceptance_passes_and_is_deterministic() -> None:
    first = evaluate_tokenizer_acceptance(_report(), "bharat-bpe", _thresholds())
    second = evaluate_tokenizer_acceptance(_report(), "bharat-bpe", _thresholds())
    assert first == second
    assert first["passed"] is True
    assert len(first["acceptance_sha256"]) == 64


@pytest.mark.parametrize(
    ("section", "key", "value", "failed_check"),
    [
        ("aggregate", "record_count", 9, "record_count"),
        ("aggregate", "unknown_token_rate", 0.01, "unknown_token_rate"),
        ("aggregate", "micro_fertility", 2.01, "micro_fertility"),
        ("round_trip", "required_pass_rate", 0.99, "required_round_trip_rate"),
        ("byte_coverage", "complete", False, "complete_byte_coverage"),
    ],
)
def test_acceptance_rejects_failed_threshold(
    section: str, key: str, value: object, failed_check: str
) -> None:
    report = copy.deepcopy(_report())
    report[section]["bharat-bpe"][key] = value  # type: ignore[index]
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    assert result["passed"] is False
    failed = {check["name"] for check in result["checks"] if not check["passed"]}
    assert failed_check in failed


def test_acceptance_rejects_language_fertility_failure() -> None:
    report = copy.deepcopy(_report())
    report["per_language"]["bharat-bpe"]["hi"]["micro_fertility"] = 2.1  # type: ignore[index]
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    language_check = next(
        check for check in result["checks"] if check["name"] == "language_micro_fertility"
    )
    assert result["passed"] is False
    assert language_check["actual"] == {"hi": 2.1}


def test_thresholds_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown threshold fields"):
        TokenizerAcceptanceThresholds.from_dict({"unexpected": 1})


def test_acceptance_requires_named_tokenizer() -> None:
    with pytest.raises(ValueError, match="is not present"):
        evaluate_tokenizer_acceptance(_report(), "missing", _thresholds())
