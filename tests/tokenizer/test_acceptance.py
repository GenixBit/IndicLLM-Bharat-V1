from __future__ import annotations

import hashlib
import json

import pytest

from bharat.tokenizer.acceptance import (
    ThresholdConfiguration,
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
        "per_script": {"bharat-bpe": {"Latin": {"record_count": 12}}},
        "per_domain": {"bharat-bpe": {"gen": {"record_count": 12}}},
        "per_category": {"bharat-bpe": {"general": {"record_count": 12}}},
        "round_trip": {
            "bharat-bpe": {
                "required_pass_rate": 1.0,
                "required_pass_count": 12,
                "canonical_pass_rate": 1.0,
                "canonical_pass_count": 12,
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


def test_missing_per_script_rejected() -> None:
    report = _realistic_report()
    del report["per_script"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluation_report(report)


def test_missing_per_domain_rejected() -> None:
    report = _realistic_report()
    del report["per_domain"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluation_report(report)


def test_missing_per_category_rejected() -> None:
    report = _realistic_report()
    del report["per_category"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluation_report(report)


def test_inconsistent_unknown_count_rate_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["unknown_token_count"] = 5  # type: ignore[index]
    report["aggregate"]["bharat-bpe"]["unknown_token_rate"] = 0.0  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="unknown_token_rate mismatch"):
        validate_evaluation_report(report)


def test_inconsistent_required_pass_count_rate_rejected() -> None:
    report = _realistic_report()
    report["round_trip"]["bharat-bpe"]["required_pass_count"] = 10  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_rate"] = 1.0  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="required_pass_rate mismatch"):
        validate_evaluation_report(report)


def test_per_language_total_mismatch_rejected() -> None:
    report = _realistic_report()
    report["per_language"]["bharat-bpe"]["en"]["record_count"] = 99  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="sum of record_count"):
        validate_evaluation_report(report)


def test_unknown_count_exceeds_token_count_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["unknown_token_count"] = 999  # type: ignore[index]
    report["aggregate"]["bharat-bpe"]["token_count"] = 500  # type: ignore[index]
    # Set rate to 1.0 (100% unknown) even though true rate is 1.998
    # The cross-field check will catch the mismatch
    report["aggregate"]["bharat-bpe"]["unknown_token_rate"] = 1.0  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="unknown_token_rate|unknown_token_count"):
        validate_evaluation_report(report)


def test_nan_fertility_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["micro_fertility"] = float("nan")  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_evaluation_report(report)


def test_inf_fertility_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["micro_fertility"] = float("inf")  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="must be a finite number"):
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
    assert "threshold_configuration_status" in result
    assert "threshold_evidence_scope" in result
    assert "threshold_configuration_sha256" in result


def test_acceptance_sha256_deterministic() -> None:
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    dig = result["acceptance_sha256"]
    assert isinstance(dig, str)
    assert len(dig) == 64
    result2 = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    assert result2["acceptance_sha256"] == dig


def test_acceptance_includes_configuration_digest() -> None:
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", _thresholds())
    cd = result["threshold_configuration_sha256"]
    assert isinstance(cd, str)
    assert len(cd) == 64


def test_provisional_metadata_changes_config_digest() -> None:
    t1 = _thresholds()
    config1 = ThresholdConfiguration(
        schema_version="tokenizer-acceptance-thresholds-v1",
        status="provisional",
        thresholds=t1,
    )
    config2 = ThresholdConfiguration(
        schema_version="tokenizer-acceptance-thresholds-v1",
        status="production",
        thresholds=t1,
    )
    r1 = evaluate_tokenizer_acceptance(
        _realistic_report(), "bharat-bpe", t1, threshold_config=config1
    )
    r2 = evaluate_tokenizer_acceptance(
        _realistic_report(), "bharat-bpe", t1, threshold_config=config2
    )
    assert r1["threshold_configuration_sha256"] != r2["threshold_configuration_sha256"]
    assert r1["acceptance_sha256"] != r2["acceptance_sha256"]


def test_unknown_config_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown configuration fields"):
        ThresholdConfiguration.from_payload(
            {
                "schema_version": "tokenizer-acceptance-thresholds-v1",
                "thresholds": {"min_record_count": 1},
                "extra_field": "x",
            }
        )


def test_acceptance_rejects_low_record_count() -> None:
    thresh = TokenizerAcceptanceThresholds(min_record_count=13)
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", thresh)
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "record_count" in failed


def test_acceptance_rejects_elevated_unknown_rate() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["unknown_token_rate"] = 0.01  # type: ignore[index]
    report["aggregate"]["bharat-bpe"]["unknown_token_count"] = 1  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "unknown_token_rate" in failed


def test_acceptance_rejects_high_fertility() -> None:
    thresh = TokenizerAcceptanceThresholds(max_micro_fertility=1.0)
    result = evaluate_tokenizer_acceptance(_realistic_report(), "bharat-bpe", thresh)
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "micro_fertility" in failed


def test_acceptance_rejects_low_round_trip_rate() -> None:
    thresh = TokenizerAcceptanceThresholds(min_required_round_trip_rate=1.0)
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["record_count"] = 100  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_rate"] = 0.99  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_count"] = 99  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["canonical_pass_count"] = 99  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["canonical_pass_rate"] = 0.99  # type: ignore[index]
    report["per_language"]["bharat-bpe"]["en"]["record_count"] = 50  # type: ignore[index]
    report["per_language"]["bharat-bpe"]["hi"]["record_count"] = 50  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    assert result["passed"] is False
    failed = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "required_round_trip_rate" in failed


def test_acceptance_byte_coverage_checks_values() -> None:
    report = _realistic_report()
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
    with pytest.raises(ValueError, match="must be a finite number"):
        TokenizerAcceptanceThresholds.from_dict(
            {
                "min_record_count": 1,
                "min_records_per_required_language": 1,
                "min_required_round_trip_rate": float("nan"),
            }
        )


def test_threshold_infinity_rejected() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
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
    thresh = TokenizerAcceptanceThresholds.from_dict({"min_record_count": 10})
    assert thresh.min_record_count == 10


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


# ── Canonical pass rate tests ───────────────────────────────────────


def test_canonical_rate_zero_fails() -> None:
    report = _realistic_report()
    report["round_trip"]["bharat-bpe"]["canonical_pass_rate"] = 0.0  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["canonical_pass_count"] = 0  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_count"] = 0  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_rate"] = 0.0  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    thresh = TokenizerAcceptanceThresholds(min_canonical_pass_rate=0.5)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    cp_check = next(c for c in result["checks"] if c["name"] == "canonical_equivalent_pass_rate")
    assert cp_check["passed"] is False


def test_canonical_rate_below_minimum_fails() -> None:
    report = _realistic_report()
    # 12 records, 10 canonical pass -> exact rate = 10/12
    report["round_trip"]["bharat-bpe"]["canonical_pass_rate"] = 10.0 / 12.0  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["canonical_pass_count"] = 10  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_count"] = 10  # type: ignore[index]
    report["round_trip"]["bharat-bpe"]["required_pass_rate"] = 10.0 / 12.0  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    thresh = TokenizerAcceptanceThresholds(min_canonical_pass_rate=0.95)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    cp_check = next(c for c in result["checks"] if c["name"] == "canonical_equivalent_pass_rate")
    assert cp_check["passed"] is False


def test_canonical_rate_meets_minimum() -> None:
    report = _realistic_report()
    thresh = TokenizerAcceptanceThresholds(min_canonical_pass_rate=0.95)
    result = evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)
    cp_check = next(c for c in result["checks"] if c["name"] == "canonical_equivalent_pass_rate")
    assert cp_check["passed"] is True


# ── NaN/Inf rejection in acceptance ─────────────────────────────────


def test_nan_overall_fertility_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["micro_fertility"] = float("nan")  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="finite"):
        evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())


def test_inf_overall_fertility_rejected() -> None:
    report = _realistic_report()
    report["aggregate"]["bharat-bpe"]["micro_fertility"] = float("inf")  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    with pytest.raises(ValueError, match="finite"):
        evaluate_tokenizer_acceptance(report, "bharat-bpe", _thresholds())


def test_nan_per_language_fertility_rejected() -> None:
    report = _realistic_report()
    report["per_language"]["bharat-bpe"]["en"]["micro_fertility"] = float("nan")  # type: ignore[index]
    report["report_sha256"] = _compute_digest(report)
    thresh = TokenizerAcceptanceThresholds(max_language_micro_fertility=5.0)
    with pytest.raises(ValueError, match="finite"):
        evaluate_tokenizer_acceptance(report, "bharat-bpe", thresh)


def test_inf_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="must be a finite number"):
        TokenizerAcceptanceThresholds(min_required_round_trip_rate=float("inf"))
