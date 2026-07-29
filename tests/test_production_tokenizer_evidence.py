from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.evaluation import compute_evaluation_dataset_sha256
from bharat.tokenizer.production_evidence import validate_production_evidence
from scripts.validate_production_tokenizer_evidence import main

_EXIT_ACCEPTED = 0
_EXIT_VALID_CANDIDATE = 1
_EXIT_INVALID = 2
_EXIT_EXISTING_OUTPUT = 3


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


def _build_bpe_tokenizer(tmp_path: Path) -> Path:
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
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    return path


def _build_input_jsonl(tmp_path: Path) -> Path:
    records = [
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
            "text": "नमस्ते भारत",
        },
        {
            "id": "rec-3",
            "language": "en",
            "script": "Latin",
            "domain": "news",
            "text": "Good morning",
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
            "min_required_round_trip_rate": 0.0,
            "min_canonical_pass_rate": 0.0,
            "max_unknown_token_rate": 1.0,
            "require_complete_byte_coverage": False,
            "required_languages": ["en"],
            "min_records_per_required_language": 1,
        },
    }
    path = tmp_path / "thresholds.json"
    path.write_bytes(_canonical(payload))
    return path


def _build_provisional_thresholds(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "tokenizer-acceptance-thresholds-v1",
        "status": "provisional",
        "evidence_scope": "synthetic-local-only",
        "notes": [],
        "thresholds": {
            "min_record_count": 1,
            "min_required_round_trip_rate": 0.0,
            "min_canonical_pass_rate": 0.0,
            "max_unknown_token_rate": 1.0,
            "require_complete_byte_coverage": False,
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
) -> Path:
    from bharat.tokenizer.evaluation import load_evaluation_records

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
                "canonical_pass_rate": None,
                "canonical_pass_count": 0,
                "canonical_evaluated_count": 0,
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
    path = tmp_path / "report.json"
    path.write_bytes(_canonical(report))
    return path


def _build_acceptance_decision(
    tmp_path: Path,
    report_path: Path,
    thresholds_path: Path,
    tokenizer_name: str,
    tokenizer_fp: str,
    passed: bool = True,
    overrides: dict | None = None,
) -> Path:
    from bharat.tokenizer.acceptance import ThresholdConfiguration, evaluate_tokenizer_acceptance

    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresh = json.loads(thresholds_path.read_text(encoding="utf-8"))
    config = ThresholdConfiguration.from_payload(thresh)
    decision = evaluate_tokenizer_acceptance(report, tokenizer_name, config)
    if overrides:
        decision.update(overrides)
    if not passed:
        decision["passed"] = False
        decision["checks"][0]["passed"] = False
        canonical = json.dumps(
            {k: v for k, v in decision.items() if k != "acceptance_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        decision["acceptance_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = tmp_path / "decision.json"
    path.write_bytes(_canonical(decision))
    return path


def _build_manifest(
    tmp_path: Path,
    tokenizer_path: Path,
    input_path: Path,
    report_path: Path,
    decision_path: Path,
    thresholds_path: Path,
    status: str = "candidate",
    tokenizer_name: str = "test-bpe",
    overrides: dict | None = None,
) -> Path:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "tokenizer-production-evidence-manifest-v1",
        "status": status,
        "evidence_scope": "production-local-approved",
        "repository_commit_sha": "a" * 40,
        "tokenizer": {
            "artifact_path": str(tokenizer_path.relative_to(tmp_path)),
            "artifact_sha256": _digest(tokenizer_path),
            "fingerprint": decision.get("tokenizer_fingerprint", ""),
            "vocab_size": 260,
            "normalization": "NFC",
            "byte_alphabet_complete": status == "accepted",
        },
        "evaluation_input": {
            "path": str(input_path.relative_to(tmp_path)),
            "sha256": _digest(input_path),
        },
        "evaluation_report": {
            "path": str(report_path.relative_to(tmp_path)),
            "sha256": _digest(report_path),
        },
        "acceptance_decision": {
            "path": str(decision_path.relative_to(tmp_path)),
            "sha256": _digest(decision_path),
        },
        "threshold_configuration": {
            "path": str(thresholds_path.relative_to(tmp_path)),
            "sha256": _digest(thresholds_path),
        },
        "language_coverage": {
            "required_languages": ["en"],
            "record_counts": {"en": 2, "hi": 1},
        },
        "generating_commands": ["test-command"],
    }
    if overrides:
        manifest.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical(manifest))
    return path


@pytest.fixture
def valid_evidence(tmp_path: Path) -> Path:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        status="candidate",
    )
    return manifest_path


@pytest.fixture
def accepted_evidence(tmp_path: Path) -> Path:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        status="accepted",
    )
    return manifest_path


# ── Candidate validation ──────────────────────────────────────────


def test_candidate_never_reports_accepted(valid_evidence: Path) -> None:
    result = validate_production_evidence(valid_evidence)
    assert result.status == "candidate"
    assert result.accepted is False
    assert result.valid is True


def test_candidate_valid_result(valid_evidence: Path) -> None:
    result = validate_production_evidence(valid_evidence)
    assert result.valid is True
    assert result.accepted is False
    assert len(result.errors) == 0


def test_noncanonical_manifest_is_rejected(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path, tokenizer_path, input_path, report_path, decision_path, thresholds_path
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = validate_production_evidence(manifest_path)
    assert "manifest: JSON bytes are not canonical" in result.errors


def test_path_escape_is_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["evaluation_input"].__setitem__("path", "../outside.jsonl"),
    )
    joined = " ".join(result.errors)
    assert "parent traversal" in joined or "escapes" in joined


def _update_and_validate(manifest_path: Path, update) -> object:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    update(payload)
    manifest_path.write_bytes(_canonical(payload))
    return validate_production_evidence(manifest_path)


def test_non_nfc_manifest_normalization_is_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("normalization", "none"),
    )
    assert "tokenizer.normalization must be NFC" in result.errors


def test_non_positive_language_count_is_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["language_coverage"]["record_counts"].__setitem__("en", 0),
    )
    assert "must be a positive integer" in " ".join(result.errors)


# ── Artifact / report / decision fingerprint binding ──────────────


def test_artifact_report_fingerprint_mismatch(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("fingerprint", "a" * 64),
    )
    assert "fingerprint" in " ".join(result.errors).lower()


def test_artifact_decision_fingerprint_mismatch(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    tok_path_str = payload["tokenizer"]["artifact_path"]
    tok_path = valid_evidence.parent / tok_path_str
    _ = BPETokenizer.load(tok_path)
    wrong_fp = hashlib.sha256(b"wrong").hexdigest()
    payload["tokenizer"]["fingerprint"] = wrong_fp
    decision_path = valid_evidence.parent / payload["acceptance_decision"]["path"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["tokenizer_fingerprint"] = wrong_fp
    decision_path.write_bytes(_canonical(decision))
    payload["acceptance_decision"]["sha256"] = _digest(decision_path)
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert "fingerprint" in " ".join(result.errors).lower()


def test_report_from_different_tokenizer(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    report_path = valid_evidence.parent / payload["evaluation_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["tokenizer_names"] = ["other-tokenizer"]
    report["tokenizer_fingerprints"] = {"other-tokenizer": "b" * 64}
    for k in (
        "aggregate",
        "per_language",
        "per_script",
        "per_domain",
        "per_category",
        "round_trip",
        "fragmentation",
        "byte_coverage",
    ):
        report[k] = {
            "other-tokenizer": next(iter(report[k].values()))
            if isinstance(report[k], dict)
            else report[k]
        }
    report["comparison"] = []
    report["failed_records"] = []
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report_path.write_bytes(_canonical(report))
    payload["evaluation_report"]["sha256"] = _digest(report_path)
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert any("not in" in e or "fingerprint" in e for e in result.errors)


# ── Evaluation input binding ──────────────────────────────────────


def test_evaluation_input_from_different_dataset(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    report_path = valid_evidence.parent / payload["evaluation_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["input_dataset_sha256"] = "a" * 64
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report_path.write_bytes(_canonical(report))
    payload["evaluation_report"]["sha256"] = _digest(report_path)
    payload["evaluation_report"]["report_sha256"] = report["report_sha256"]
    payload["evaluation_report"]["input_dataset_sha256"] = "a" * 64
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert any("dataset digest" in e for e in result.errors)


def test_dataset_metadata_change_detected(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    input_path = valid_evidence.parent / payload["evaluation_input"]["path"]
    lines = input_path.read_text(encoding="utf-8").strip().split("\n")
    record = json.loads(lines[0])
    record["text"] = "Modified text"
    lines[0] = json.dumps(record, sort_keys=True, ensure_ascii=False)
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["evaluation_input"]["sha256"] = _digest(input_path)
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert any("dataset digest" in e for e in result.errors)


# ── Manifest schema validation ────────────────────────────────────


def test_invalid_fingerprint_format(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("fingerprint", "xyz"),
    )
    assert "SHA-256" in " ".join(result.errors)


def test_boolean_vocab_size_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("vocab_size", True),
    )
    assert "boolean" in " ".join(result.errors).lower()


def test_boolean_byte_alphabet_complete_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("byte_alphabet_complete", "yes"),
    )
    assert "boolean" in " ".join(result.errors).lower()


def test_windows_absolute_path_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("artifact_path", "C:\\Windows\\bad.json"),
    )
    assert "Windows drive path" in " ".join(result.errors)


def test_backslash_traversal_rejected(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["tokenizer"].__setitem__("artifact_path", "..\\..\\etc\\passwd"),
    )
    assert "backslash traversal" in " ".join(result.errors)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    import tempfile

    outside_dir = Path(tempfile.mkdtemp())
    outside = outside_dir / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    symlink_path = tmp_path / "evil_link.json"
    os.symlink(str(outside), str(symlink_path))
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        overrides={
            "tokenizer": {
                "artifact_path": "evil_link.json",
                "artifact_sha256": _digest(symlink_path),
                "fingerprint": tokenizer_fp,
                "vocab_size": 260,
                "normalization": "NFC",
                "byte_alphabet_complete": False,
            }
        },
    )
    result = validate_production_evidence(manifest_path)
    assert any("symlink" in e for e in result.errors)


def test_invalid_language_identifier(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["language_coverage"]["required_languages"].__setitem__(0, "en-US!"),
    )
    assert "does not match" in " ".join(result.errors)


def test_missing_required_language_count(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["language_coverage"]["required_languages"].__setitem__(0, "fr"),
    )
    assert "missing required language" in " ".join(result.errors)


def test_invalid_extra_record_count(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["language_coverage"]["record_counts"].__setitem__("invalid!", 5),
    )
    assert "does not match" in " ".join(result.errors)


# ── Production threshold alignment ────────────────────────────────


def test_accepted_requires_production_thresholds(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_provisional_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        status="accepted",
    )
    result = validate_production_evidence(manifest_path)
    assert any("production thresholds" in e for e in result.errors)


def test_count_below_threshold_minimum(valid_evidence: Path) -> None:
    result = _update_and_validate(
        valid_evidence,
        lambda p: p["language_coverage"]["record_counts"].__setitem__("en", 0),
    )
    assert "positive integer" in " ".join(result.errors)


# ── Accepted evidence invariants ──────────────────────────────────


def test_accepted_evidence_is_valid(accepted_evidence: Path) -> None:
    result = validate_production_evidence(accepted_evidence)
    assert result.valid is True
    assert result.accepted is True
    assert len(result.errors) == 0


def test_accepted_evidence_requires_passing_decision(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=False
    )
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        status="accepted",
    )
    result = validate_production_evidence(manifest_path)
    joined = " ".join(result.errors)
    assert any(
        msg in joined
        for msg in ("passing acceptance decision", "does not match recomputed decision")
    )


# ── CLI behavior ──────────────────────────────────────────────────


def test_cli_refuses_overwrite(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path, tokenizer_path, input_path, report_path, decision_path, thresholds_path
    )
    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")
    assert main([str(manifest_path), "--output", str(output)]) == _EXIT_EXISTING_OUTPUT
    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_output_byte_verification(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path, tokenizer_path, input_path, report_path, decision_path, thresholds_path
    )
    output = tmp_path / "result.json"
    rc = main([str(manifest_path), "--output", str(output)])
    assert rc in (_EXIT_ACCEPTED, _EXIT_VALID_CANDIDATE)
    assert output.exists()
    written = output.read_bytes()
    from bharat.tokenizer.production_evidence import validate_production_evidence

    expected = validate_production_evidence(manifest_path).canonical_bytes()
    assert written == expected


def test_cli_exit_codes(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path, tokenizer_path, input_path, report_path, decision_path, thresholds_path
    )
    output = tmp_path / "result.json"
    rc = main([str(manifest_path), "--output", str(output)])
    from bharat.tokenizer.production_evidence import validate_production_evidence

    result = validate_production_evidence(manifest_path)
    if result.accepted:
        assert rc == _EXIT_ACCEPTED
    elif result.valid:
        assert rc == _EXIT_VALID_CANDIDATE
    else:
        assert rc == _EXIT_INVALID


def test_invalid_evidence_exit_code(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not json", encoding="utf-8")
    rc = main([str(manifest)])
    assert rc == _EXIT_INVALID


def test_pre_existing_output_preserved(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path, tokenizer_path, input_path, report_path, decision_path, thresholds_path
    )
    output = tmp_path / "result.json"
    output.write_text("original", encoding="utf-8")
    rc = main([str(manifest_path), "--output", str(output)])
    assert rc == _EXIT_EXISTING_OUTPUT
    assert output.read_text(encoding="utf-8") == "original"


def test_valid_candidate_result(tmp_path: Path) -> None:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_production_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _build_evaluation_report(tmp_path, "test-bpe", tokenizer_fp, input_path)
    decision_path = _build_acceptance_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp, passed=True
    )
    manifest_path = _build_manifest(
        tmp_path,
        tokenizer_path,
        input_path,
        report_path,
        decision_path,
        thresholds_path,
        status="candidate",
    )
    result = validate_production_evidence(manifest_path)
    assert result.valid is True
    assert result.accepted is False


def test_invalid_candidate_result(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    result = validate_production_evidence(manifest_path)
    assert result.valid is False
    assert result.accepted is False


def test_changing_text_changes_digest_and_invalidates(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    input_path = valid_evidence.parent / payload["evaluation_input"]["path"]
    original_text = input_path.read_text(encoding="utf-8")
    new_text = original_text.replace("Hello world", "Hola mundo")
    input_path.write_text(new_text, encoding="utf-8")
    payload["evaluation_input"]["sha256"] = _digest(input_path)
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert any("dataset digest" in e for e in result.errors)


def test_changing_language_changes_digest_and_invalidates(valid_evidence: Path) -> None:
    payload = json.loads(valid_evidence.read_text(encoding="utf-8"))
    input_path = valid_evidence.parent / payload["evaluation_input"]["path"]
    lines = input_path.read_text(encoding="utf-8").strip().split("\n")
    record = json.loads(lines[0])
    record["language"] = "fr"
    lines[0] = json.dumps(record, sort_keys=True, ensure_ascii=False)
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["evaluation_input"]["sha256"] = _digest(input_path)
    valid_evidence.write_bytes(_canonical(payload))
    result = validate_production_evidence(valid_evidence)
    assert any("dataset digest" in e for e in result.errors)
