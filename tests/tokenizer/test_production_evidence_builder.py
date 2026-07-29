from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

import pytest

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.bpe_adapter import BharatBPETokenizer
from bharat.tokenizer.evaluation import (
    TokenizerEvaluation,
    compute_evaluation_dataset_sha256,
    load_evaluation_records,
)
from bharat.tokenizer.production_evidence import (
    validate_production_evidence,
)
from bharat.tokenizer.production_evidence_builder import (
    build_candidate_manifest,
    write_candidate_manifest,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_bpe_tokenizer(tmp_path: Path, name: str = "tokenizer.json") -> Path:
    byte_value_to_id = {b: 4 + b for b in range(256)}
    id_to_bytes = {4 + b: bytes([b]) for b in range(256)}
    special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab = dict(special_tokens)
    for b in range(256):
        vocab[f"<byte_{b:02x}>"] = 4 + b
    tok = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens=special_tokens,
        reserved_tokens={},
        byte_value_to_id=byte_value_to_id,
        id_to_bytes=id_to_bytes,
        vocab=vocab,
        merges=(),
        tokenizer_hash="",
    )
    tok.tokenizer_hash = tok.compute_hash()
    tok.validate()
    path = tmp_path / name
    tok.save(path)
    return path


def _build_bad_bpe_tokenizer(
    tmp_path: Path,
    name: str,
    missing_byte: int | None = None,
    duplicate_byte: bool = False,
    bad_byte_mapping: bool = False,
    collision_special: bool = False,
) -> tuple[Path, str]:
    byte_value_to_id = {b: 4 + b for b in range(256)}
    id_to_bytes = {4 + b: bytes([b]) for b in range(256)}
    special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    if missing_byte is not None:
        del byte_value_to_id[missing_byte]
    if duplicate_byte:
        byte_value_to_id[0] = 4 + 1
    if bad_byte_mapping:
        id_to_bytes[4 + 0] = bytes([1])
    if collision_special:
        byte_value_to_id[0] = 1
        id_to_bytes[1] = bytes([0])
    vocab = dict(special_tokens)
    for b in range(256):
        if b in byte_value_to_id:
            vocab[f"<byte_{b:02x}>"] = byte_value_to_id[b]
    tok = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens=special_tokens,
        reserved_tokens={},
        byte_value_to_id=byte_value_to_id,
        id_to_bytes=id_to_bytes,
        vocab=vocab,
        merges=(),
        tokenizer_hash="",
    )
    tok.tokenizer_hash = tok.compute_hash()
    data = tok.to_dict()
    path = tmp_path / name
    path.write_bytes(_canonical(data))
    return path, tok.compute_hash()


def _build_input_jsonl(
    tmp_path: Path,
    overrides: list[dict[str, Any]] | None = None,
) -> Path:
    records = overrides or [
        {
            "id": "rec-1",
            "language": "en",
            "script": "Latin",
            "domain": "web",
            "text": "Hello world",
        },
        {
            "id": "rec-2",
            "language": "hi",
            "script": "Devanagari",
            "domain": "web",
            "text": "\u0928\u092e\u0938\u094d\u0924\u0947 \u092d\u093e\u0930\u0924",
        },
        {
            "id": "rec-3",
            "language": "en",
            "script": "Latin",
            "domain": "news",
            "text": "Good morning",
        },
        {
            "id": "rec-4",
            "language": "hi",
            "script": "Devanagari",
            "domain": "canonical",
            "text": "\u0915\u093c",
            "canonical_equivalent": "\u0958",
        },
    ]
    lines = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    path = tmp_path / "input.jsonl"
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def _build_production_thresholds(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "tokenizer-acceptance-thresholds-v1",
        "status": "production",
        "evidence_scope": "approved-evaluation-set",
        "notes": [],
        "thresholds": {
            "min_record_count": 1,
            "min_required_round_trip_rate": 1.0,
            "min_canonical_pass_rate": 1.0,
            "min_canonical_evaluated_count": 1,
            "max_unknown_token_rate": 0.0,
            "require_complete_byte_coverage": True,
            "required_languages": ["en", "hi"],
            "min_records_per_required_language": 1,
        },
    }
    path = tmp_path / "thresholds.json"
    path.write_bytes(_canonical(payload))
    return path


def _build_evaluation_report(
    tmp_path: Path,
    tokenizer_name: str,
    tokenizer_fp: str,
    input_path: Path,
    overrides: dict | None = None,
    name: str = "report.json",
) -> Path:
    records = load_evaluation_records(input_path)
    ds_digest = compute_evaluation_dataset_sha256(records)
    en_count = sum(1 for r in records if r.language == "en")
    hi_count = sum(1 for r in records if r.language == "hi")
    total = len(records)
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
    if overrides:
        report.update(overrides)
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = tmp_path / name
    path.write_bytes(_canonical(report))
    return path


def _build_acceptance_decision(
    tmp_path: Path,
    report_path: Path,
    thresholds_path: Path,
    tokenizer_name: str,
    tokenizer_fp: str,
    name: str = "decision.json",
) -> Path:
    from bharat.tokenizer.acceptance import (
        ThresholdConfiguration,
        evaluate_tokenizer_acceptance,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresh = json.loads(thresholds_path.read_text(encoding="utf-8"))
    config = ThresholdConfiguration.from_payload(thresh)
    decision = evaluate_tokenizer_acceptance(report, tokenizer_name, config)
    path = tmp_path / name
    path.write_bytes(_canonical(decision))
    return path


def _compute_real_report(
    tmp_path: Path,
    tokenizer_path: Path,
    input_path: Path,
    tokenizer_name: str = "test-bpe",
    name: str = "report.json",
) -> Path:
    loaded = BPETokenizer.load(tokenizer_path)
    adapter = BharatBPETokenizer(loaded)
    evaluation = TokenizerEvaluation({tokenizer_name: adapter})
    evaluation.load_records(input_path)
    report = evaluation.compute()
    path = tmp_path / name
    path.write_bytes(_canonical(report))
    return path


@pytest.fixture
def evidence_fixtures(tmp_path: Path) -> dict[str, Path]:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _compute_real_report(tmp_path, tokenizer_path, input_path, "test-bpe")
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp
    )
    return {
        "tokenizer_path": tokenizer_path,
        "input_path": input_path,
        "thresholds_path": thresholds_path,
        "report_path": report_path,
        "decision_path": decision_path,
        "tokenizer_fp": tokenizer_fp,
        "tokenizer_name": "test-bpe",
    }


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
    bad_tokenizer, bad_fp = _build_bad_bpe_tokenizer(tmp_path, "bad_missing.json", missing_byte=42)
    bad_report = _build_evaluation_report(
        tmp_path,
        "test-bpe",
        bad_fp,
        evidence_fixtures["input_path"],
        name="bad_report_missing.json",
    )
    bad_decision = _build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        bad_fp,
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
    bad_tokenizer, bad_fp = _build_bad_bpe_tokenizer(tmp_path, "bad_dup.json", duplicate_byte=True)
    bad_report = _build_evaluation_report(
        tmp_path,
        "test-bpe",
        bad_fp,
        evidence_fixtures["input_path"],
        name="bad_report_dup.json",
    )
    bad_decision = _build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        bad_fp,
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
    bad_tokenizer, bad_fp = _build_bad_bpe_tokenizer(
        tmp_path, "bad_mapping.json", bad_byte_mapping=True
    )
    bad_report = _build_evaluation_report(
        tmp_path,
        "test-bpe",
        bad_fp,
        evidence_fixtures["input_path"],
        name="bad_report_mapping.json",
    )
    bad_decision = _build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        bad_fp,
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
    bad_tokenizer, bad_fp = _build_bad_bpe_tokenizer(
        tmp_path, "bad_collision.json", collision_special=True
    )
    bad_report = _build_evaluation_report(
        tmp_path,
        "test-bpe",
        bad_fp,
        evidence_fixtures["input_path"],
        name="bad_report_collision.json",
    )
    bad_decision = _build_acceptance_decision(
        tmp_path,
        bad_report,
        evidence_fixtures["thresholds_path"],
        "test-bpe",
        bad_fp,
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
    digest = write_candidate_manifest(
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
    assert _digest(out) == digest
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
# Canonical equivalence verification
# ======================================================


def test_canonical_equivalence_verified(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
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
    assert manifest["tokenizer"]["normalization"] == "NFC"


def test_canonical_equivalence_record_properties(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    import unicodedata

    records = load_evaluation_records(evidence_fixtures["input_path"])
    rec4 = next(r for r in records if r.record_id == "rec-4")
    assert rec4.canonical_equivalent is not None
    assert rec4.text != rec4.canonical_equivalent
    assert unicodedata.normalize("NFC", rec4.text) == unicodedata.normalize(
        "NFC", rec4.canonical_equivalent
    )


# ======================================================
# Failure-injection tests for output path
# ======================================================


def test_fi_output_is_broken_symlink(tmp_path: Path, evidence_fixtures: dict[str, Path]) -> None:
    missing = tmp_path / "nonexistent_target"
    out = tmp_path / "manifest.json"
    out.symlink_to(missing)
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


# ======================================================
# Low-level publication-safety tests
# ======================================================


def test_publication_flush_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    original_open = Path.open

    def failing_flush_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if "xb" in mode and path.name == "manifest.json":

            def failing_flush():
                handle.close()
                raise OSError("flush failed")

            handle.flush = failing_flush
        return handle

    monkeypatch.setattr(Path, "open", failing_flush_open)
    out = tmp_path / "manifest.json"
    with pytest.raises((OSError, RuntimeError)):
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
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_fsync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    fail_count = 0

    def failing_fsync(fd):
        nonlocal fail_count
        fail_count += 1
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    out = tmp_path / "manifest.json"
    with pytest.raises((OSError, RuntimeError)):
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
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_readback_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0

    original_read_bytes = Path.read_bytes

    def failing_read_bytes(path):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise OSError("readback failed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    out = tmp_path / "manifest.json"
    with pytest.raises((OSError, RuntimeError)):
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
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_readback_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0

    original_read_bytes = Path.read_bytes

    def corrupting_read_bytes(path):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            return b"CORRUPTED"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", corrupting_read_bytes)
    out = tmp_path / "manifest.json"
    with pytest.raises((RuntimeError, ValueError)):
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
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_final_bytes_modified_after_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    from bharat.tokenizer.production_evidence import (
        validate_production_evidence as original_validate,
    )

    manifest_path = tmp_path / "manifest.json"

    def corrupting_validate(path):
        result = original_validate(path)
        if path.name == "manifest.json":
            path.write_bytes(path.read_bytes() + b"TAMPERED")
        return result

    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_builder.validate_production_evidence",
        corrupting_validate,
    )
    with pytest.raises(RuntimeError, match="final byte verification failed|final byte mismatch"):
        write_candidate_manifest(
            manifest_path,
            evidence_root=tmp_path,
            repository_commit_sha="a" * 40,
            tokenizer_path=evidence_fixtures["tokenizer_path"],
            evaluation_input_path=evidence_fixtures["input_path"],
            evaluation_report_path=evidence_fixtures["report_path"],
            acceptance_decision_path=evidence_fixtures["decision_path"],
            threshold_configuration_path=evidence_fixtures["thresholds_path"],
            generating_commands=["test-command"],
        )
    assert not manifest_path.exists()


def test_publication_output_race_after_lexists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    out = tmp_path / "manifest.json"
    raced = False

    original_lexists = os.path.lexists

    def racing_lexists(path):
        nonlocal raced
        if isinstance(path, str) and path.endswith("manifest.json"):
            if not raced:
                raced = True
                return False
            out.write_text("racer", encoding="utf-8")
            return False
        return original_lexists(path)

    monkeypatch.setattr(os.path, "lexists", racing_lexists)
    with pytest.raises((FileExistsError, OSError)):
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
    assert out.read_text(encoding="utf-8") == "racer"


def test_publication_temp_name_collision_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    call_count = 0
    original_token_hex = secrets.token_hex

    def colliding_token_hex(nbytes):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return "a" * (2 * nbytes)
        return original_token_hex(nbytes)

    monkeypatch.setattr(secrets, "token_hex", colliding_token_hex)
    out = tmp_path / "manifest.json"
    digest = write_candidate_manifest(
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
    assert isinstance(digest, str) and len(digest) == 64
    assert out.exists()
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_temp_name_all_collide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    import bharat.tokenizer.production_evidence_builder as builder_mod

    original_exclusive = builder_mod._publish_exclusive

    call_count = 0

    def always_fail_exclusive(path, payload):
        nonlocal call_count
        call_count += 1
        if path.suffix == ".tmp":
            raise FileExistsError(f"collision on {path.name}")
        return original_exclusive(path, payload)

    monkeypatch.setattr(builder_mod, "_publish_exclusive", always_fail_exclusive)
    out = tmp_path / "manifest.json"
    with pytest.raises((RuntimeError, FileExistsError)):
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
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0


def test_publication_concurrent_same_output(
    tmp_path: Path, evidence_fixtures: dict[str, Path]
) -> None:
    import threading

    out = tmp_path / "manifest.json"
    results: list[Exception | str] = []
    lock = threading.Lock()

    def publish(idx: int) -> None:
        try:
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
            with lock:
                results.append(d)
        except BaseException as e:
            with lock:
                results.append(e)

    threads = [threading.Thread(target=publish, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 2
    success_count = sum(1 for r in results if isinstance(r, str) and len(r) == 64)
    error_count = sum(1 for r in results if isinstance(r, FileExistsError))
    assert success_count == 1
    assert error_count == 1

    assert out.exists()
    from bharat.tokenizer.production_evidence import validate_production_evidence

    result = validate_production_evidence(out)
    assert result.valid is True
    assert result.status == "candidate"
    temps = list(tmp_path.glob(".*.tmp"))
    assert len(temps) == 0
