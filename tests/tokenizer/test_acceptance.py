from __future__ import annotations

import copy
import json

import pytest

from bharat.tokenizer.acceptance import (
    TokenizerAcceptanceThresholds,
    evaluate_tokenizer_acceptance,
)
from bharat.tokenizer.evaluation import validate_evaluation_report

# ── Helpers ──────────────────────────────────────────────────────────


def _minimal_valid_report() -> dict[str, object]:
    return {
        "schema_version": "eval-v1",
        "evaluator_version": "1.0.3",
        "report_sha256": "a" * 64,
        "input_dataset_sha256": "b" * 64,
        "tokenizer_names": ["bharat-bpe"],
        "tokenizer_fingerprints": {"bharat-bpe": "fp123"},
        "aggregate": {
            "bharat-bpe": {
                "record_count": 12,
                "token_count": 100,
                "unknown_token_count": 0,
                "unknown_token_rate": 0.0,
                "micro_fertility": 1.25,
                "macro_fertility": 1.25,
            }
        },
        "per_language": {
            "bharat-bpe": {
                "en": {"micro_fertility": 1.0, "record_count": 6},
                "hi": {"micro_fertility": 1.5, "record_count": 6},
            }
        },
        "round_trip": {
            "bharat-bpe": {
                "required_pass_rate": 1.0,
                "required_pass_count": 12,
                "canonical_pass_rate": 1.0,
            }
        },
        "byte_coverage": {
            "bharat-bpe": {
                "status": "complete",
                "complete": True,
                "reachable_count": 256,
                "missing_byte_values": [],
            }
        },
        "fragmentation": {"bharat-bpe": {}},
        "comparison": [],
        "failed_records": [],
    }


def _realistic_report() -> dict[str, object]:
    r = _minimal_valid_report()
    r["report_sha256"] = _compute_digest(r)
    return r


def _compute_digest(report: dict) -> str:
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _thresholds() -> TokenizerAcceptanceThresholds:
    return TokenizerAcceptanceThresholds(
        min_record_count=10,
        min_required_round_trip_rate=1.0,
        max_unknown_token_rate=0.0,
        require_complete_byte_coverage=True,
        max_micro_fertility=2.0,
        max_language_micro_fertility=2.0,
    )


# ── validate_evaluation_report tests ─────────────────────────────────


def test_valid_report_accepted() -> None:
    report = _realistic_report()
    # should not raise
    validate_evaluation_report(report)


def test_modified_aggregate_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["record_count"] = 999  # type: ignore[index]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_report(report)


def test_modified_round_trip_rejected() -> None:
    report = _realistic_report()
    report["round_trip"]["bharat-bpe"]["required_pass_rate"] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_report(report)


def test_modified_byte_coverage_rejected() -> None:
    report = _realistic_report()
    report["byte_coverage"]["bharat-bpe"]["complete"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_report(report)


def test_missing_digest_rejected() -> None:
    report = _minimal_valid_report()
    report.pop("report_sha256", None)
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluation_report(report)


def test_malformed_digest_rejected() -> None:
    report = _realistic_report()
    report["report_sha256"] = "not-a-hex-string"
    with pytest.raises(ValueError, match="lowercase 64-character hex"):
        validate_evaluation_report(report)


def test_unsupported_schema_rejected() -> None:
    report = _realistic_report()
    report["schema_version"] = "eval-v0"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        validate_evaluation_report(report)


def test_invalid_input_dataset_sha256() -> None:
    report = _realistic_report()
    report["input_dataset_sha256"] = "zzz"
    with pytest.raises(ValueError, match="lowercase 64-character hex"):
        validate_evaluation_report(report)


def test_duplicate_tokenizer_names_rejected() -> None:
    report = _realistic_report()
    report["tokenizer_names"] = ["tok", "tok"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_evaluation_report(report)


def test_empty_tokenizer_names_rejected() -> None:
    report = _realistic_report()
    report["tokenizer_names"] = []
    with pytest.raises(ValueError, match="non-empty"):
        validate_evaluation_report(report)


def test_missing_tokenizer_fingerprint_rejected() -> None:
    report = _realistic_report()
    report["tokenizer_fingerprints"] = {}
    with pytest.raises(ValueError, match="missing or empty fingerprint"):
        validate_evaluation_report(report)


def test_aggregate_record_count_type_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["record_count"] = "12"  # type: ignore[index]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_report(report)


# ── Acceptance gate tests ───────────────────────────────────────────


def test_acceptance_passes_and_is_deterministic() -> None:
    first = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    second = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    assert first == second
    assert first["passed"] is True
    assert len(first["acceptance_sha256"]) == 64


def test_acceptance_includes_provenance() -> None:
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    assert result["tokenizer_name"] == "bharat-bpe"
    assert result["tokenizer_fingerprint"] == "fp123"
    assert result["input_dataset_sha256"] == "b" * 64
    assert result["input_report_sha256"] == result["input_report_sha256"]
    assert result["evaluation_schema_version"] == "eval-v1"
    assert result["evaluator_version"] == "1.0.3"
    assert result["threshold_schema_version"] == "tokenizer-acceptance-thresholds-v1"
    assert "thresholds" in result
    assert "thresholds_sha256" in result


def test_acceptance_sha256_deterministic() -> None:
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    dig = result["acceptance_sha256"]
    assert isinstance(dig, str)
    assert len(dig) == 64
    # Re-run with same inputs = same digest
    result2 = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    assert result2["acceptance_sha256"] == dig


@pytest.mark.parametrize(
    ("section", "key", "value", "failed_check"),
    [
        ("aggregate", "record_count", 9, "record_count"),
        ("aggregate", "unknown_token_rate", 0.01, "unknown_token_rate"),
        ("aggregate", "micro_fertility", 2.01, "micro_fertility"),
        ("round_trip", "required_pass_rate", 0.99, "required_round_trip_rate"),
    ],
)
def test_acceptance_rejects_failed_threshold(
    section: str, key: str, value: object, failed_check: str
) -> None:
    report = copy.deepcopy(_realistic_report())
    report[section]["bharat-bpe"][key] = value  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    assert result["passed"] is False
    failed = {check["name"] for check in result["checks"] if not check["passed"]}
    assert failed_check in failed


def test_acceptance_byte_coverage_checks_values() -> None:
    report = _realistic_report()
    # Test with byte coverage set to unavailable
    report["byte_coverage"]["bharat-bpe"] = {  # type: ignore[index]
        "status": "unavailable",
        "complete": False,
        "reachable_count": 0,
        "missing_byte_values": [],
    }
    report["report_sha256"] = _compute_digest(report)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    assert result["passed"] is False
    bc_check = next(c for c in result["checks"] if c["name"] == "complete_byte_coverage")
    assert bc_check["passed"] is False


def test_acceptance_byte_coverage_complete_passes() -> None:
    report = _realistic_report()
    report["byte_coverage"]["bharat-bpe"] = {  # type: ignore[index]
        "status": "complete",
        "complete": True,
        "reachable_count": 256,
        "missing_byte_values": [],
    }
    report["report_sha256"] = _compute_digest(report)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    assert result["passed"] is True


def test_acceptance_rejects_language_fertility_failure() -> None:
    report = _realistic_report()
    report["per_language"]["bharat-bpe"]["hi"]["micro_fertility"] = 2.1  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
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
        evaluate_tokenizer_acceptance(_realistic_report(), "missing", _thresholds())


def test_acceptance_rejects_unsupported_evaluator_version() -> None:
    report = _realistic_report()
    report["evaluator_version"] = "0.0.1"
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="unsupported evaluator_version"):
        evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())


def test_threshold_string_rejected_for_int_field() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        TokenizerAcceptanceThresholds.from_dict({"min_record_count": "abc"})


def test_threshold_bool_rejected_for_int_field() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        TokenizerAcceptanceThresholds.from_dict({"min_record_count": True})


def test_threshold_nan_rejected() -> None:
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        TokenizerAcceptanceThresholds.from_dict(
            {
                "min_record_count": 1,
                "min_records_per_required_language": 1,
                "min_required_round_trip_rate": float("nan"),
            }
        )


def test_threshold_infinity_rejected() -> None:
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        TokenizerAcceptanceThresholds.from_dict(
            {
                "min_record_count": 1,
                "min_records_per_required_language": 1,
                "min_required_round_trip_rate": float("inf"),
            }
        )


def test_threshold_unknown_top_level_rejected() -> None:
    with pytest.raises(ValueError, match="unknown threshold fields"):
        TokenizerAcceptanceThresholds.from_dict({"min_record_count": 5, "unknown_extra": "x"})


def test_threshold_none_int_field_uses_default() -> None:
    thresh = TokenizerAcceptanceThresholds.from_dict({"min_record_count": None})
    assert thresh.min_record_count == 1


def test_required_languages_missing() -> None:
    thresh = TokenizerAcceptanceThresholds(required_languages=["fr", "de"])
    report = _realistic_report()
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    lang_check = next(c for c in result["checks"] if c["name"] == "required_languages_present")
    assert lang_check["passed"] is False
    assert "fr" in str(lang_check["actual"])
    assert "de" in str(lang_check["actual"])


def test_required_languages_present() -> None:
    thresh = TokenizerAcceptanceThresholds(required_languages=["en", "hi"])
    report = _realistic_report()
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    lang_check = next(c for c in result["checks"] if c["name"] == "required_languages_present")
    assert lang_check["passed"] is True


def test_acceptance_includes_thresholds_sha256() -> None:
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    ts = result["thresholds_sha256"]
    assert isinstance(ts, str)
    assert len(ts) == 64


def test_changing_threshold_changes_digests() -> None:
    t1 = TokenizerAcceptanceThresholds(max_micro_fertility=2.0)
    t2 = TokenizerAcceptanceThresholds(max_micro_fertility=3.0)
    r1 = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", t1)
    r2 = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", t2)
    assert r1["thresholds_sha256"] != r2["thresholds_sha256"]
    assert r1["acceptance_sha256"] != r2["acceptance_sha256"]
