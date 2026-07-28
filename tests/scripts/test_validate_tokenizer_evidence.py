from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from scripts.validate_tokenizer_evidence import validate_evidence

_EVIDENCE_DIR = Path("evidence/tokenizer/milestone-6-1-synthetic")
_MANIFEST_PATH = _EVIDENCE_DIR / "manifest.json"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_evidence_to(tmp_path: Path) -> Path:
    # Replicate the exact nesting so that manifest_path.parents[3] == tmp_path
    dest = tmp_path / "evidence" / "tokenizer" / "milestone-6-1-synthetic"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ["manifest.json", "evaluation-report.json", "acceptance-decision.json"]:
        shutil.copy2(str(_EVIDENCE_DIR / fname), str(dest / fname))
    return dest / "manifest.json"


def _copy_fixtures_to(tmp_path: Path) -> None:
    fixture_src = Path("tests/fixtures/tokenizer_eval/all.jsonl")
    fixture_dst = tmp_path / "tests/fixtures/tokenizer_eval/all.jsonl"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(fixture_src), str(fixture_dst))

    tok_src = Path("tests/fixtures/tiny_bpe_tokenizer.json")
    tok_dst = tmp_path / "tests/fixtures/tiny_bpe_tokenizer.json"
    tok_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(tok_src), str(tok_dst))

    config_src = Path("configs/tokenizers/bpe-64k-acceptance.json")
    config_dst = tmp_path / "configs/tokenizers/bpe-64k-acceptance.json"
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(config_src), str(config_dst))


# ── Valid pack ───────────────────────────────────────────────────


def test_valid_pack(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


# ── Decision recomputation ───────────────────────────────────────


def test_decision_recomputation(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


def test_unrelated_decision_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    decision_file = p.parent / "acceptance-decision.json"
    decision = _read_json(decision_file)
    decision["passed"] = not decision["passed"]
    decision["checks"][0]["passed"] = not decision["checks"][0]["passed"]
    del decision["acceptance_sha256"]
    canonical = json.dumps(decision, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    decision["acceptance_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(decision_file, decision)
    m["acceptance_decision"]["sha256"] = hashlib.sha256(decision_file.read_bytes()).hexdigest()
    m["acceptance_decision"]["passed"] = decision["passed"]
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("recomputed" in e for e in errors)


# ── Digest tampering ─────────────────────────────────────────────


def test_report_sha256_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_decision_sha256_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_fingerprint_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["fingerprint"] = "0" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("fingerprint" in e for e in errors)


def test_input_report_digest_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["report_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("report_sha256 mismatch" in e for e in errors)


def test_input_dataset_digest_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["input_dataset_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("input_dataset_sha256 mismatch" in e for e in errors)


def test_thresholds_sha256_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["threshold_configuration"]["thresholds_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("thresholds SHA-256 mismatch" in e for e in errors)


def test_configuration_sha256_mismatch(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["threshold_configuration"]["configuration_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("configuration SHA-256 mismatch" in e for e in errors)


# ── Path traversal ───────────────────────────────────────────────


def test_path_traversal_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["artifact_path"] = "../outside.txt"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("escapes" in e.lower() for e in errors)


def test_multi_level_traversal_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["artifact_path"] = "../../../../etc/passwd"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("escapes" in e.lower() or "outside" in e.lower() for e in errors)


def test_symlink_escape_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    ev_dir = p.parent
    m = _read_json(p)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    symlink_path = ev_dir / "evil_link.json"
    os.symlink(str(outside), str(symlink_path))
    m["tokenizer"]["artifact_path"] = "evil_link.json"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("symlink" in e.lower() or "escapes" in e.lower() for e in errors)


def test_symlink_outside_evidence_in_report_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    ev_dir = p.parent
    m = _read_json(p)
    outside = tmp_path / "outside_report.json"
    outside.write_text('{"fake": true}', encoding="utf-8")
    symlink_path = ev_dir / "evil_report_link.json"
    os.symlink(str(outside), str(symlink_path))
    m["evaluation_report"]["path"] = "evil_report_link.json"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("symlink" in e.lower() or "escapes" in e.lower() for e in errors)


def test_valid_contained_path_succeeds(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


# ── Strict schema rejection ──────────────────────────────────────


def test_unknown_field_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["unknown_key"] = "value"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("unknown" in e for e in errors)


def test_nan_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    report_file = p.parent / "evaluation-report.json"
    text = report_file.read_text(encoding="utf-8")
    text = text.replace("0.0", "NaN")
    report_file.write_text(text, encoding="utf-8")
    errors = validate_evidence(p)
    assert any("NaN" in e for e in errors)


def test_infinity_rejected(tmp_path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    report_file = p.parent / "evaluation-report.json"
    text = report_file.read_text(encoding="utf-8")
    text = text.replace("0.0", "Infinity")
    report_file.write_text(text, encoding="utf-8")
    errors = validate_evidence(p)
    assert any("Infinity" in e for e in errors)


# ── Missing file ─────────────────────────────────────────────────


def test_missing_manifest_file() -> None:
    errors = validate_evidence(Path("/nonexistent/manifest.json"))
    assert any("not found" in e for e in errors)


# ── Deterministic double generation ──────────────────────────────


def test_deterministic_double_generation(tmp_path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_tokenizer_evidence"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stdout:{result.stdout}\nstderr:{result.stderr}"
    assert "byte-identical" in result.stdout


def test_generated_vs_committed_byte_equality(tmp_path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_tokenizer_evidence", "--verify-committed"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stdout:{result.stdout}\nstderr:{result.stderr}"
    assert "matches committed" in result.stdout
