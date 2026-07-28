from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_tokenizer_evidence import validate_evidence

_EVIDENCE_DIR = Path("evidence/tokenizer/milestone-6-1-synthetic")
_MANIFEST_PATH = _EVIDENCE_DIR / "manifest.json"


def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(payload: dict) -> None:
    _MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _restore_manifest() -> None:
    _write_manifest(_load_manifest())


# ── Fixture ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _preserve_manifest() -> None:
    original = _MANIFEST_PATH.read_bytes()
    yield
    _MANIFEST_PATH.write_bytes(original)


# ── Valid pack ───────────────────────────────────────────────────────


def test_valid_pack() -> None:
    errors = validate_evidence(_MANIFEST_PATH)
    assert errors == []


# ── Tokenizer artifact tampering ─────────────────────────────────────


def test_tokenizer_artifact_sha256_mismatch() -> None:
    m = _load_manifest()
    m["tokenizer"]["artifact_sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_tokenizer_fingerprint_mismatch() -> None:
    m = _load_manifest()
    m["tokenizer"]["fingerprint"] = "bad" * 16
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("fingerprint mismatch" in e for e in errors)


# ── Evaluation fixture tampering ─────────────────────────────────────


def test_evaluation_fixture_sha256_mismatch() -> None:
    m = _load_manifest()
    m["evaluation_fixture"]["sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("SHA-256 mismatch" in e for e in errors)


# ── Report tampering ─────────────────────────────────────────────────


def test_evaluation_report_sha256_mismatch() -> None:
    m = _load_manifest()
    m["evaluation_report"]["sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("SHA-256 mismatch" in e for e in errors)


# ── Threshold configuration tampering ────────────────────────────────


def test_threshold_configuration_sha256_mismatch() -> None:
    m = _load_manifest()
    m["threshold_configuration"]["sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_thresholds_sha256_mismatch() -> None:
    m = _load_manifest()
    m["threshold_configuration"]["thresholds_sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("thresholds SHA-256 mismatch" in e for e in errors)


def test_configuration_sha256_mismatch() -> None:
    m = _load_manifest()
    m["threshold_configuration"]["configuration_sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("configuration SHA-256 mismatch" in e for e in errors)


# ── Acceptance decision tampering ────────────────────────────────────


def test_acceptance_decision_sha256_mismatch() -> None:
    m = _load_manifest()
    m["acceptance_decision"]["sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("SHA-256 mismatch" in e for e in errors)


# ── Mismatched fingerprints ──────────────────────────────────────────


def test_mismatched_fingerprint_in_manifest() -> None:
    m = _load_manifest()
    m["tokenizer"]["fingerprint"] = "0" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("fingerprint" in e for e in errors)


# ── Mismatched dataset digest ────────────────────────────────────────


def test_mismatched_input_dataset_digest() -> None:
    m = _load_manifest()
    m["evaluation_report"]["input_dataset_sha256"] = "a" * 64
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("input_dataset_sha256 mismatch" in e for e in errors)


# ── Missing file ─────────────────────────────────────────────────────


def test_missing_manifest_file() -> None:
    errors = validate_evidence(Path("/nonexistent/manifest.json"))
    assert any("not found" in e for e in errors)


# ── Absolute-path rejection ──────────────────────────────────────────


def test_absolute_tokenizer_path_rejected() -> None:
    m = _load_manifest()
    m["tokenizer"]["artifact_path"] = "/etc/passwd"
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("must be relative" in e for e in errors)


def test_absolute_evaluation_fixture_path_rejected() -> None:
    m = _load_manifest()
    m["evaluation_fixture"]["path"] = "/etc/passwd"
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("must be relative" in e for e in errors)


# ── Status rejection ─────────────────────────────────────────────────


def test_production_status_rejected() -> None:
    m = _load_manifest()
    m["status"] = "production"
    _write_manifest(m)
    errors = validate_evidence(_MANIFEST_PATH)
    assert any("status must be provisional" in e for e in errors)


# ── Offline execution ────────────────────────────────────────────────


def test_validator_does_not_access_network() -> None:
    errors = validate_evidence(_MANIFEST_PATH)
    assert errors == []
