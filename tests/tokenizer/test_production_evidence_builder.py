from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.production_evidence import (
    ProductionEvidenceValidation,
    validate_production_evidence,
)
from bharat.tokenizer.production_evidence_builder import (
    build_candidate_manifest,
    write_candidate_manifest,
)
from tests.tokenizer.evidence_fixtures import (
    build_acceptance_decision,
    build_bad_bpe_tokenizer,
    canonical_bytes,
    digest,
)


def _canonical(value: object) -> bytes:
    return canonical_bytes(value)


def _digest(path: Path) -> str:
    return digest(path)


def _tamper_report(report_path: Path, key_path: list[str], new_value: Any) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = report
    for key in key_path[:-1]:
        target = target[key]
    target[key_path[-1]] = new_value
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    forged = report_path.with_name("forged_" + report_path.name)
    forged.write_bytes(_canonical(report))
    return forged


def _make_minimal_report(
    tmp_path: Path,
    input_path: Path,
    tokenizer_name: str,
    tokenizer_fp: str,
    name: str = "report.json",
) -> Path:
    records = json.loads(
        "[" + input_path.read_text(encoding="utf-8").strip().replace("\n", ",") + "]"
    )
    en_count = sum(1 for r in records if r["language"] == "en")
    hi_count = sum(1 for r in records if r["language"] == "hi")
    total = len(records)
    from bharat.tokenizer.evaluation import (
        compute_evaluation_dataset_sha256,
        load_evaluation_records,
    )

    loaded = load_evaluation_records(input_path)
    ds_digest = compute_evaluation_dataset_sha256(loaded)
    report = {
        "schema_version": "eval-v1",
        "evaluator_version": "1.0.3",
        "input_dataset_sha256": ds_digest,
        "tokenizer_names": [tokenizer_name],
        "tokenizer_fingerprints": {tokenizer_name: tokenizer_fp},
        "aggregate": {
            tokenizer_name: {
                "record_count": total,
                "char_count": 50,
                "byte_count": 100,
                "token_count": 100,
                "unknown_token_count": 0,
                "unknown_token_rate": 0.0,
                "records_with_unknown": 0,
                "special_token_count": 0,
                "byte_token_count": 100,
                "merged_token_count": 0,
                "micro_fertility": 2.0,
                "macro_fertility": 2.0,
                "min_fertility": 1.0,
                "max_fertility": 3.0,
                "median_fertility": 2.0,
            }
        },
        "per_language": {
            tokenizer_name: {
                "en": {
                    "record_count": en_count,
                    "char_count": 20,
                    "byte_count": 40,
                    "token_count": 40,
                    "micro_fertility": 2.0,
                    "macro_fertility": 2.0,
                    "min_fertility": 1.0,
                    "max_fertility": 3.0,
                    "median_fertility": 2.0,
                },
                "hi": {
                    "record_count": hi_count,
                    "char_count": 30,
                    "byte_count": 60,
                    "token_count": 60,
                    "micro_fertility": 2.0,
                    "macro_fertility": 2.0,
                    "min_fertility": 1.0,
                    "max_fertility": 3.0,
                    "median_fertility": 2.0,
                },
            }
        },
        "per_script": {tokenizer_name: {}},
        "per_domain": {tokenizer_name: {}},
        "per_category": {tokenizer_name: {}},
        "round_trip": {
            tokenizer_name: {
                "required_pass_rate": 1.0,
                "required_pass_count": total,
                "canonical_pass_rate": 1.0,
                "canonical_pass_count": total,
                "canonical_evaluated_count": total,
                "exact_pass_count": total,
                "exact_pass_rate": 1.0,
                "nfc_pass_count": total,
                "nfc_pass_rate": 1.0,
                "failure_records": [],
            }
        },
        "fragmentation": {tokenizer_name: {}},
        "byte_coverage": {
            tokenizer_name: {
                "status": "complete",
                "complete": True,
                "reachable_count": 256,
                "missing_byte_values": [],
            }
        },
        "comparison": [],
        "failed_records": [],
        "report_sha256": "0" * 64,
    }
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = tmp_path / name
    path.write_bytes(_canonical(report))
    return path


# -- 1. Valid candidate manifest --


def test_valid_candidate_manifest(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["status"] == "candidate"
    assert manifest["schema_version"] == "tokenizer-production-evidence-manifest-v1"
    assert list(manifest["language_coverage"]["required_languages"]) == sorted(
        manifest["language_coverage"]["record_counts"]
    )


# -- 2. Canonical deterministic output --


def test_canonical_deterministic(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    m1 = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    m2 = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert m1 == m2
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


# -- 3. Two runs produce byte-identical payloads --


def test_byte_identical_runs(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    d1 = write_candidate_manifest(
        out1,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    d2 = write_candidate_manifest(
        out2,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert d1 == d2
    assert out1.read_bytes() == out2.read_bytes()


# -- 4. All paths are correctly relative --


def test_relative_paths(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert not Path(manifest["tokenizer"]["artifact_path"]).is_absolute()
    assert not Path(manifest["evaluation_input"]["path"]).is_absolute()
    assert not Path(manifest["evaluation_report"]["path"]).is_absolute()
    assert not Path(manifest["acceptance_decision"]["path"]).is_absolute()
    assert not Path(manifest["threshold_configuration"]["path"]).is_absolute()


# -- 5. Output-location contract --


def test_output_in_root_succeeds(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    d = write_candidate_manifest(
        out,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert isinstance(d, str) and len(d) == 64


def test_output_outside_root_fails(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    root_sub = tmp_path / "evidence"
    root_sub.mkdir(exist_ok=True)
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="output must be directly inside"):
        write_candidate_manifest(
            out,
            evidence_root=root_sub,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_nested_output_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    nested = tmp_path / "subdir" / "manifest.json"
    nested.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="output must be directly inside"):
        write_candidate_manifest(
            nested,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


# -- 6. Tokenizer fingerprint binding --


def test_tokenizer_fingerprint_binding(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["tokenizer"]["fingerprint"] == evidence_fixtures["tokenizer_fp"]


# -- 7. Vocabulary-size binding --


def test_vocab_size_binding(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    tokenizer = BPETokenizer.load(evidence_fixtures["tokenizer_path"])
    assert manifest["tokenizer"]["vocab_size"] == tokenizer.vocab_size


# -- 8. Evaluation dataset binding --


def test_evaluation_dataset_binding(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["evaluation_input"]["sha256"] == _digest(evidence_fixtures["input_path"])


# -- 9. Report-to-decision binding --


def test_report_to_decision_binding(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["evaluation_report"]["sha256"] == _digest(evidence_fixtures["report_path"])
    assert manifest["acceptance_decision"]["sha256"] == _digest(evidence_fixtures["decision_path"])


# -- 10. Threshold-configuration binding --


def test_threshold_configuration_binding(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["threshold_configuration"]["sha256"] == _digest(
        evidence_fixtures["thresholds_path"]
    )


# -- 11. Per-language count derivation --


def test_per_language_count_derivation(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["language_coverage"]["record_counts"]["en"] == 2
    assert manifest["language_coverage"]["record_counts"]["hi"] == 2


# -- 12. Incomplete byte alphabet rejection --


def test_missing_byte_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    bad_tokenizer, bad_fp = build_bad_bpe_tokenizer(tmp_path, "bad_missing.json", missing_byte=42)
    bad_report = _make_minimal_report(
        tmp_path,
        evidence_fixtures["input_path"],
        "test-bpe",
        bad_fp,
        name="bad_report_missing.json",
    )
    bad_decision = build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        name="bad_decision_missing.json",
    )
    with pytest.raises((ValueError, OSError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=bad_tokenizer,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report,
            acceptance_decision_path=bad_decision,
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_duplicate_byte_id_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    bad_tokenizer, bad_fp = build_bad_bpe_tokenizer(tmp_path, "bad_dup.json", duplicate_byte=True)
    bad_report = _make_minimal_report(
        tmp_path,
        evidence_fixtures["input_path"],
        "test-bpe",
        bad_fp,
        name="bad_report_dup.json",
    )
    bad_decision = build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        name="bad_decision_dup.json",
    )
    with pytest.raises((ValueError, OSError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=bad_tokenizer,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report,
            acceptance_decision_path=bad_decision,
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_bad_id_to_bytes_mapping_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    bad_tokenizer, bad_fp = build_bad_bpe_tokenizer(
        tmp_path, "bad_mapping.json", bad_byte_mapping=True
    )
    bad_report = _make_minimal_report(
        tmp_path,
        evidence_fixtures["input_path"],
        "test-bpe",
        bad_fp,
        name="bad_report_mapping.json",
    )
    bad_decision = build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        name="bad_decision_mapping.json",
    )
    with pytest.raises((ValueError, OSError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=bad_tokenizer,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report,
            acceptance_decision_path=bad_decision,
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_collision_with_special_token_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    bad_tokenizer, bad_fp = build_bad_bpe_tokenizer(
        tmp_path, "bad_collision.json", collision_special=True
    )
    bad_report = _make_minimal_report(
        tmp_path,
        evidence_fixtures["input_path"],
        "test-bpe",
        bad_fp,
        name="bad_report_collision.json",
    )
    bad_decision = build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        name="bad_decision_collision.json",
    )
    with pytest.raises((ValueError, OSError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=bad_tokenizer,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report,
            acceptance_decision_path=bad_decision,
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_valid_byte_alphabet_accepted(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    manifest = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert manifest["tokenizer"]["byte_alphabet_complete"] is True


# -- 13. Non-finite JSON rejection --


def test_non_finite_json_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    bad_report_path = tmp_path / "bad_nan_report.json"
    bad_report_path.write_text(json.dumps({"value": float("nan")}), encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report_path,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_infinity_json_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    bad_report_path = tmp_path / "bad_inf_report.json"
    bad_report_path.write_text(json.dumps({"value": float("inf")}), encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report_path,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


# -- 14. Malformed UTF-8 rejection --


def test_malformed_utf8_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    bad_report_path = tmp_path / "bad_utf8_report.json"
    bad_report_path.write_bytes(b"\xff\xfe{'invalid}")
    with pytest.raises((OSError, ValueError, UnicodeDecodeError)):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=bad_report_path,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


# -- 15. Existing output preservation --


def test_existing_output_preserved(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    out.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert out.read_text(encoding="utf-8") == "original"


# -- 16. Publication rollback --


def test_publication_rollback(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    digest_val = write_candidate_manifest(
        out,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert out.exists()
    assert _digest(out) == digest_val
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


# -- 17. Generated manifest passes validation --


def test_generated_manifest_validates(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    write_candidate_manifest(
        out,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    result = validate_production_evidence(out)
    assert result.valid is True
    assert result.status == "candidate"
    assert result.accepted is False


# -- 18. Manifest can be moved with its evidence --


def test_manifest_movable_with_evidence(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    write_candidate_manifest(
        out,
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    result = validate_production_evidence(out)
    assert result.valid is True


# -- 19. Different repo SHA changes digest --


def test_different_sha_produces_different_digest(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    m1 = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="a" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    m2 = build_candidate_manifest(
        evidence_root=tmp_path,
        repository_commit_sha="b" * 40,
        tokenizer_path=evidence_fixtures["tokenizer_path"],
        evaluation_input_path=evidence_fixtures["input_path"],
        evaluation_report_path=evidence_fixtures["report_path"],
        acceptance_decision_path=evidence_fixtures["decision_path"],
        threshold_configuration_path=evidence_fixtures["thresholds_path"],
        generating_commands=["test-command"],
    )
    assert m1 != m2
    p1 = _canonical(m1)
    p2 = _canonical(m2)
    assert p1 != p2


# ======================================================
# Forged-report rejection tests (9 metric fields)
# ======================================================


def test_forged_micro_fertility_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["aggregate", "test-bpe", "micro_fertility"],
        99.9,
    )
    with pytest.raises(ValueError, match="does not match report recomputed"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_unknown_token_rate_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["aggregate", "test-bpe", "unknown_token_rate"],
        0.5,
    )
    with pytest.raises(ValueError, match="unknown_token_rate mismatch"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_required_pass_rate_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["round_trip", "test-bpe", "required_pass_rate"],
        0.5,
    )
    with pytest.raises(ValueError, match="required_pass_rate mismatch"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_canonical_evaluated_count_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["round_trip", "test-bpe", "canonical_evaluated_count"],
        999,
    )
    with pytest.raises(ValueError, match="canonical_pass_rate mismatch"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_canonical_pass_rate_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["round_trip", "test-bpe", "canonical_pass_rate"],
        0.0,
    )
    with pytest.raises(ValueError, match="canonical_pass_rate mismatch"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_byte_coverage_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["byte_coverage", "test-bpe", "status"],
        "incomplete",
    )
    with pytest.raises(ValueError, match="does not match report recomputed"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_fragmentation_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["fragmentation", "test-bpe"],
        {"en": {"fragility": 0.5}},
    )
    with pytest.raises(ValueError, match="does not match report recomputed"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_per_language_metrics_rejected(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["per_language", "test-bpe", "en", "micro_fertility"],
        99.9,
    )
    with pytest.raises(ValueError, match="does not match report recomputed"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_forged_failed_records_rejected(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    forged = _tamper_report(
        evidence_fixtures["report_path"],
        ["failed_records"],
        [{"id": "fake-failure", "reason": "injected"}],
    )
    with pytest.raises(ValueError, match="does not match report recomputed"):
        build_candidate_manifest(
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


# ======================================================
# Genuine rollback tests (13 failure modes)
# ======================================================


def _assert_cleanup(root: Path, output: Path, exception_type: type) -> None:
    with pytest.raises(exception_type):
        write_candidate_manifest(
            output,
            evidence_root=root,
            repository_commit_sha="a" * 40,
            tokenizer_path=root / "tokenizer.json",
            evaluation_input_path=root / "input.jsonl",
            evaluation_report_path=root / "report.json",
            acceptance_decision_path=root / "decision.json",
            threshold_configuration_path=root / "thresholds.json",
            generating_commands=["test-command"],
        )
    assert not output.exists()
    orphans = list(root.glob("*.*.tmp"))
    assert len(orphans) == 0, f"orphaned temp files: {orphans}"


def test_rollback_output_exists(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("preexisting", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_candidate_manifest(
            output,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=tmp_path / "nonexistent.json",
            evaluation_input_path=tmp_path / "nonexistent.jsonl",
            evaluation_report_path=tmp_path / "nonexistent.json",
            acceptance_decision_path=tmp_path / "nonexistent.json",
            threshold_configuration_path=tmp_path / "nonexistent.json",
            generating_commands=["test-command"],
        )
    assert output.read_text(encoding="utf-8") == "preexisting"
    assert len(list(tmp_path.glob("*.*.tmp"))) == 0


def test_rollback_invalid_repo_sha(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="repository_commit_sha"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="invalid-sha",
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_empty_commands(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="generating_commands"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=[],
        )
    assert not out.exists()


def test_rollback_non_existent_tokenizer(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    out = tmp_path / "manifest.json"
    missing = tmp_path / "no_such_tokenizer.json"
    with pytest.raises((ValueError, OSError)):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=missing,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_tampered_report(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    forged = _tamper_report(evidence_fixtures["report_path"], ["schema_version"], "eval-v2")
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=forged,
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_tampered_decision(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    decision = json.loads(evidence_fixtures["decision_path"].read_text(encoding="utf-8"))
    decision["passed"] = not decision["passed"]
    decision_canonical = json.dumps(
        decision, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    decision_digest = hashlib.sha256(decision_canonical.encode("utf-8")).hexdigest()
    decision["acceptance_sha256"] = decision_digest
    forged = tmp_path / "forged_decision.json"
    forged.write_bytes(_canonical(decision))
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="does not match recomputed decision"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=forged,
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_wrong_tokenizer_fingerprint(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    from bharat.tokenizer.bpe import BPETokenizer

    distinct_byte_value_to_id = {b: 260 + b for b in range(256)}
    distinct_id_to_bytes = {260 + b: bytes([b]) for b in range(256)}
    distinct_special = {"<pad>": 256, "<unk>": 257, "<bos>": 258, "<eos>": 259}
    distinct_vocab = dict(distinct_special)
    for b in range(256):
        distinct_vocab[f"<byte_{b:02x}>"] = 260 + b
    tok = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens=distinct_special,
        reserved_tokens={},
        byte_value_to_id=distinct_byte_value_to_id,
        id_to_bytes=distinct_id_to_bytes,
        vocab=distinct_vocab,
        merges=(),
        tokenizer_hash="",
    )
    tok.tokenizer_hash = tok.compute_hash()
    tok.validate()
    second_tokenizer = tmp_path / "distinct_tokenizer.json"
    tok.save(second_tokenizer)

    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="tokenizer_fingerprint"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=second_tokenizer,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_temp_publication_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0
    import bharat.tokenizer.production_evidence_builder as builder_mod

    original = builder_mod._publish_exclusive

    def failing_publish(path: Path, payload: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("disk full on temp")
        original(path, payload)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder._publish_exclusive", failing_publish
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(OSError, match="disk full on temp"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()
    temps = list(tmp_path.glob(".*.*.tmp"))
    assert len(temps) == 0, f"orphaned temps: {temps}"


def test_rollback_temp_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    def failing_validation(path: Path) -> ProductionEvidenceValidation:
        if path.name.startswith("."):
            return ProductionEvidenceValidation(
                manifest_sha256="0" * 64,
                status="invalid",
                valid=False,
                accepted=False,
                errors=("mock temp validation failure",),
            )
        return validate_production_evidence(path)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder.validate_production_evidence",
        failing_validation,
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="candidate evidence validation failed"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_output_publication_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0
    import bharat.tokenizer.production_evidence_builder as builder_mod

    original = builder_mod._publish_exclusive

    def failing_publish(path: Path, payload: bytes) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("disk full on output")
        original(path, payload)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder._publish_exclusive", failing_publish
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(OSError, match="disk full on output"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_output_sha_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    import bharat.tokenizer.production_evidence_builder as builder_mod

    original = builder_mod._publish_exclusive

    def corrupting_publish(path: Path, payload: bytes) -> None:
        if path.name == "manifest.json":
            original(path, payload + b"CORRUPT")
        else:
            original(path, payload)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder._publish_exclusive", corrupting_publish
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="byte-verification failed|SHA-256 mismatch"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_post_write_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0

    def failing_validation(path: Path) -> ProductionEvidenceValidation:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ProductionEvidenceValidation(
                manifest_sha256="0" * 64,
                status="invalid",
                valid=False,
                accepted=False,
                errors=("mock post-write validation failure",),
            )
        return validate_production_evidence(path)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder.validate_production_evidence",
        failing_validation,
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="published candidate evidence validation failed"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_post_write_status_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0

    def wrong_status(path: Path) -> ProductionEvidenceValidation:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ProductionEvidenceValidation(
                manifest_sha256="0" * 64,
                status="production",
                valid=True,
                accepted=False,
                errors=(),
            )
        return validate_production_evidence(path)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder.validate_production_evidence",
        wrong_status,
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="status is 'production'"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


def test_rollback_post_write_accepted_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0

    def wrong_accepted(path: Path) -> ProductionEvidenceValidation:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ProductionEvidenceValidation(
                manifest_sha256="0" * 64,
                status="candidate",
                valid=True,
                accepted=True,
                errors=(),
            )
        return validate_production_evidence(path)

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder.validate_production_evidence",
        wrong_accepted,
    )
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="must not report accepted=True"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not out.exists()


# ======================================================
# Failure-injection tests (10 filesystem scenarios)
# ======================================================


def test_fi_output_is_directory(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    out = tmp_path / "manifest.json"
    out.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite existing output"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert out.is_dir()


def test_fi_output_is_symlink_to_file(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    target = tmp_path / "target.json"
    target.write_text("real target", encoding="utf-8")
    out = tmp_path / "manifest.json"
    out.symlink_to(target)
    with pytest.raises(FileExistsError, match="refusing to overwrite existing output"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert out.is_symlink()


def test_fi_output_is_symlink_to_dir(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    target = tmp_path / "target_dir"
    target.mkdir()
    out = tmp_path / "manifest.json"
    out.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite existing output"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert out.is_symlink()


def test_fi_tokenizer_is_dir(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    tokenizer_dir = tmp_path / "tokenizer_is_dir"
    tokenizer_dir.mkdir()
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="tokenizer_path does not exist or is not a file"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=tokenizer_dir,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_fi_tokenizer_is_broken_symlink(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    missing = tmp_path / "nonexistent.json"
    tokenizer_link = tmp_path / "broken_tokenizer_link.json"
    tokenizer_link.symlink_to(missing)
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="tokenizer_path does not exist or is not a file"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=tokenizer_link,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_fi_input_symlink_outside_root(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink_outside_input.jsonl"
    symlink.symlink_to(outside)
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="evaluation_input_path must be inside evidence root"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=symlink,
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_fi_tokenizer_symlink_outside_root(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    outside = tmp_path.parent / "outside_tokenizer.json"
    outside.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink_outside_tokenizer.json"
    symlink.symlink_to(outside)
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="tokenizer_path must be inside evidence root"):
        write_candidate_manifest(
            out,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=symlink,
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_fi_evidence_root_is_file(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    root_file = tmp_path / "root_file"
    root_file.write_text("not a directory", encoding="utf-8")
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="evidence_root must be an existing directory"):
        write_candidate_manifest(
            out,
            evidence_root=root_file,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )


def test_fi_evidence_root_missing(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    missing_root = tmp_path / "does_not_exist"
    out = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="evidence_root must be an existing directory"):
        write_candidate_manifest(
            out,
            evidence_root=missing_root,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
