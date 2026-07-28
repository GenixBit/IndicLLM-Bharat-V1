from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.validate_tokenizer_evidence import validate_evidence

_EVIDENCE_DIR = Path("evidence/tokenizer/milestone-6-1-synthetic")
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_evidence_to(tmp_path: Path) -> Path:
    dest = tmp_path / "evidence" / "tokenizer" / "milestone-6-1-synthetic"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ["manifest.json", "evaluation-report.json", "acceptance-decision.json"]:
        shutil.copy2(str(_EVIDENCE_DIR / fname), str(dest / fname))
    return dest / "manifest.json"


def _copy_fixtures_to(tmp_path: Path) -> None:
    for src in [
        "tests/fixtures/tokenizer_eval/all.jsonl",
        "tests/fixtures/tiny_bpe_tokenizer.json",
        "configs/tokenizers/bpe-64k-acceptance.json",
    ]:
        dst = tmp_path / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(Path(src)), str(dst))


# ── Valid pack ───────────────────────────────────────────────────


def test_valid_pack(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


# ── Decision recomputation ───────────────────────────────────────


def test_decision_recomputation(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


def test_unrelated_decision_rejected(tmp_path: Path) -> None:
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
    m["acceptance_decision"]["acceptance_sha256"] = decision["acceptance_sha256"]
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("recomputed" in e for e in errors)


# ── Manifest acceptance field binding ────────────────────────────


def test_manifest_acceptance_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["acceptance_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("acceptance_sha256" in e for e in errors)


def test_manifest_tokenizer_name_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["tokenizer_name"] = "wrong-name"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("tokenizer_name" in e for e in errors)


def test_manifest_input_report_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["input_report_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("input_report_sha256" in e for e in errors)


def test_manifest_tokenizer_fingerprint_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["tokenizer_fingerprint"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("tokenizer_fingerprint" in e for e in errors)


def test_manifest_passed_flag_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["passed"] = not m["acceptance_decision"]["passed"]
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("passed" in e for e in errors)


# ── Cross-field binding ──────────────────────────────────────────


def test_tokenizer_fingerprint_cross_field(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["tokenizer_fingerprint"] = "b" * 64
    m["tokenizer"]["fingerprint"] = "c" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("tokenizer_fingerprint" in e for e in errors)


# ── Digest tampering ─────────────────────────────────────────────


def test_report_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_decision_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["acceptance_decision"]["sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_fingerprint_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["fingerprint"] = "0" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("fingerprint" in e for e in errors)


def test_input_report_digest_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["report_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("report_sha256 mismatch" in e for e in errors)


def test_input_dataset_digest_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["evaluation_report"]["input_dataset_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("input_dataset_sha256 mismatch" in e for e in errors)


def test_thresholds_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["threshold_configuration"]["thresholds_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("thresholds SHA-256 mismatch" in e for e in errors)


def test_configuration_sha256_mismatch(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["threshold_configuration"]["configuration_sha256"] = "a" * 64
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("configuration SHA-256 mismatch" in e for e in errors)


# ── Path traversal ───────────────────────────────────────────────


def test_path_traversal_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["artifact_path"] = "../outside.txt"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("escapes" in e.lower() for e in errors)


def test_multi_level_traversal_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["artifact_path"] = "../../../../etc/passwd"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("escapes" in e.lower() or "outside" in e.lower() for e in errors)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
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


def test_symlink_outside_evidence_in_report_rejected(tmp_path: Path) -> None:
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


def test_valid_contained_path_succeeds(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    errors = validate_evidence(p)
    assert errors == [], errors


# ── Strict schema rejection ──────────────────────────────────────


def test_unknown_field_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    m = _read_json(p)
    m["tokenizer"]["unknown_key"] = "value"
    _write_json(p, m)
    errors = validate_evidence(p)
    assert any("unknown" in e for e in errors)


# ── Strict JSON: NaN / Infinity ──────────────────────────────────


def test_nan_dict_value_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    report_file = p.parent / "evaluation-report.json"
    text = report_file.read_text(encoding="utf-8")
    pos = text.index("0.0")
    text = text[:pos] + "NaN" + text[pos + 3 :]
    report_file.write_text(text, encoding="utf-8")
    errors = validate_evidence(p)
    assert any("NaN" in e for e in errors)


def test_nan_with_whitespace_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    manifest = _read_json(p)
    manifest["extra_value"] = "will be before NaN"
    _write_json(p, manifest)
    report_file = p.parent / "evaluation-report.json"
    report_file.write_text('{"value": NaN}', encoding="utf-8")
    errors = validate_evidence(p)
    assert any("NaN" in e for e in errors)


def test_infinity_array_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    report_file = p.parent / "evaluation-report.json"
    report_file.write_text("[Infinity]", encoding="utf-8")
    errors = validate_evidence(p)
    assert any("Infinity" in e for e in errors)


def test_neg_infinity_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    report_file = p.parent / "evaluation-report.json"
    report_file.write_text('{"value": -Infinity}', encoding="utf-8")
    errors = validate_evidence(p)
    assert any("Infinity" in e for e in errors)


def test_nested_non_finite_rejected(tmp_path: Path) -> None:
    p = _copy_evidence_to(tmp_path)
    _copy_fixtures_to(tmp_path)
    decision_file = p.parent / "acceptance-decision.json"
    decision_file.write_text('{"nested": {"inner": NaN}}', encoding="utf-8")
    errors = validate_evidence(p)
    assert any("NaN" in e for e in errors)


# ── Missing file ─────────────────────────────────────────────────


def test_missing_manifest_file() -> None:
    errors = validate_evidence(Path("/nonexistent/manifest.json"))
    assert any("not found" in e for e in errors)


# ── Deterministic double generation ──────────────────────────────


def test_deterministic_double_generation() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_tokenizer_evidence"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stdout:{result.stdout}\nstderr:{result.stderr}"
    assert "byte-identical" in result.stdout


def test_modified_run2_report_detected(tmp_path: Path) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run1_dir = tmp / "run1"
        run2_dir = tmp / "run2"
        from scripts.generate_tokenizer_evidence import _generate

        run1 = _generate(run1_dir)
        run2 = _generate(run2_dir)
        run2["report_path"].write_text(run2["report_path"].read_text() + "\n", encoding="utf-8")
        from scripts.generate_tokenizer_evidence import _compare_byte

        errors: list[str] = []
        _compare_byte("evaluation report", run1["report_path"], run2["report_path"], errors)
        assert errors, "modified run-2 report was not detected"


def test_modified_run2_decision_detected(tmp_path: Path) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run1_dir = tmp / "run1"
        run2_dir = tmp / "run2"
        from scripts.generate_tokenizer_evidence import _generate

        run1 = _generate(run1_dir)
        run2 = _generate(run2_dir)
        run2["decision_path"].write_text(run2["decision_path"].read_text() + "\n", encoding="utf-8")
        from scripts.generate_tokenizer_evidence import _compare_byte

        errors = []
        _compare_byte("acceptance decision", run1["decision_path"], run2["decision_path"], errors)
        assert errors, "modified run-2 decision was not detected"


# ── Verify committed ─────────────────────────────────────────────


def test_verify_committed_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_tokenizer_evidence", "--verify-committed"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"stdout:{result.stdout}\nstderr:{result.stderr}"
    assert "matches committed" in result.stdout


def test_modified_committed_report_detected(tmp_path: Path) -> None:
    committed_report = _EVIDENCE_DIR / "evaluation-report.json"
    orig = committed_report.read_bytes()
    try:
        committed_report.write_bytes(orig + b"\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_tokenizer_evidence", "--verify-committed"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0
    finally:
        committed_report.write_bytes(orig)


def test_modified_committed_decision_detected(tmp_path: Path) -> None:
    committed_decision = _EVIDENCE_DIR / "acceptance-decision.json"
    orig = committed_decision.read_bytes()
    try:
        committed_decision.write_bytes(orig + b"\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_tokenizer_evidence", "--verify-committed"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0
    finally:
        committed_decision.write_bytes(orig)


def test_modified_committed_manifest_detected(tmp_path: Path) -> None:
    committed_manifest = _EVIDENCE_DIR / "manifest.json"
    orig = committed_manifest.read_bytes()
    try:
        cm = json.loads(orig)
        cm["evaluation_report"]["sha256"] = "a" * 64
        committed_manifest.write_text(json.dumps(cm, indent=2), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_tokenizer_evidence", "--verify-committed"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0
    finally:
        committed_manifest.write_bytes(orig)


# ── Committed evidence files unchanged during tests ──────────────


def test_committed_files_unchanged_after_tests(tmp_path: Path) -> None:
    for fname in ["manifest.json", "evaluation-report.json", "acceptance-decision.json"]:
        f = _EVIDENCE_DIR / fname
        assert f.is_file(), f"committed file missing: {f}"
