from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bharat.tokenizer.production_evidence import validate_production_evidence
from scripts.validate_production_tokenizer_evidence import main


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


def _manifest(root: Path) -> Path:
    for name, value in {
        "tokenizer.json": b"{}",
        "input.jsonl": b"{}\n",
        "report.json": _canonical({}),
        "decision.json": _canonical({"passed": False}),
        "thresholds.json": _canonical({}),
    }.items():
        (root / name).write_bytes(value)
    payload = {
        "schema_version": "tokenizer-production-evidence-manifest-v1",
        "status": "candidate",
        "evidence_scope": "production-local-approved",
        "repository_commit_sha": "a" * 40,
        "tokenizer": {
            "artifact_path": "tokenizer.json",
            "artifact_sha256": _digest(root / "tokenizer.json"),
            "fingerprint": "b" * 64,
            "vocab_size": 1,
            "normalization": "NFC",
            "byte_alphabet_complete": False,
        },
        "evaluation_input": {
            "path": "input.jsonl",
            "sha256": _digest(root / "input.jsonl"),
        },
        "evaluation_report": {
            "path": "report.json",
            "sha256": _digest(root / "report.json"),
        },
        "acceptance_decision": {
            "path": "decision.json",
            "sha256": _digest(root / "decision.json"),
        },
        "threshold_configuration": {
            "path": "thresholds.json",
            "sha256": _digest(root / "thresholds.json"),
        },
        "language_coverage": {
            "required_languages": ["hi"],
            "record_counts": {"hi": 1},
        },
        "generating_commands": ["offline-command"],
    }
    path = root / "manifest.json"
    path.write_bytes(_canonical(payload))
    return path


def _update_manifest(path: Path, update: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert callable(update)
    update(payload)
    path.write_bytes(_canonical(payload))


def test_candidate_never_reports_accepted(tmp_path: Path) -> None:
    result = validate_production_evidence(_manifest(tmp_path))
    assert result.status == "candidate"
    assert result.accepted is False


def test_noncanonical_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result = validate_production_evidence(manifest)
    assert "manifest: JSON bytes are not canonical" in result.errors


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _update_manifest(
        manifest,
        lambda payload: payload["evaluation_input"].__setitem__(
            "path", "../outside.jsonl"
        ),
    )
    result = validate_production_evidence(manifest)
    assert "evaluation_input.path: path escapes evidence root" in result.errors


def test_non_nfc_manifest_normalization_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _update_manifest(
        manifest,
        lambda payload: payload["tokenizer"].__setitem__("normalization", "none"),
    )
    result = validate_production_evidence(manifest)
    assert "tokenizer.normalization must be NFC" in result.errors


def test_non_positive_language_count_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _update_manifest(
        manifest,
        lambda payload: payload["language_coverage"]["record_counts"].__setitem__(
            "hi", 0
        ),
    )
    result = validate_production_evidence(manifest)
    assert "record count for 'hi' must be a positive integer" in result.errors


def test_accepted_manifest_requires_independent_byte_verification(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    def mark_accepted(payload: dict[str, object]) -> None:
        payload["status"] = "accepted"
        tokenizer = payload["tokenizer"]
        assert isinstance(tokenizer, dict)
        tokenizer["byte_alphabet_complete"] = True

    _update_manifest(manifest, mark_accepted)
    result = validate_production_evidence(manifest)
    assert (
        "accepted evidence requires independently verified byte coverage" in result.errors
    )


def test_cli_refuses_overwrite(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")
    assert main([str(manifest), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == "existing"
