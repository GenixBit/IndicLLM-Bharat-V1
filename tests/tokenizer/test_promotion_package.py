from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_package as module


def _report(ready: bool = True) -> SimpleNamespace:
    data = {
        "schema_version": "tokenizer-production-evidence-readiness-v1",
        "manifest_sha256": "abc",
        "manifest_status": "candidate",
        "structurally_valid": True,
        "accepted": False,
        "ready_for_human_promotion": ready,
        "blockers": [] if ready else ["blocked"],
    }
    return SimpleNamespace(
        ready_for_human_promotion=ready,
        to_canonical_dict=lambda: data,
    )


def _package(tmp_path: Path, report: SimpleNamespace) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.json"
    readiness = tmp_path / "readiness.json"
    decision = tmp_path / "decision.json"
    manifest.write_text('{"status":"candidate"}', encoding="utf-8")
    readiness_bytes = json.dumps(
        report.to_canonical_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    readiness.write_bytes(readiness_bytes)
    decision.write_text(
        json.dumps(
            {
                "schema_version": "tokenizer-promotion-decision-v1",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "readiness_sha256": hashlib.sha256(readiness_bytes).hexdigest(),
                "decision": "approve",
                "operator": "reviewer",
                "rationale": "verified offline",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest, readiness, decision


def test_verifies_exact_approved_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    manifest, readiness, decision = _package(tmp_path, report)
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)

    result = module.verify_promotion_package(manifest, readiness, decision)

    assert result.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result.readiness_sha256 == hashlib.sha256(readiness.read_bytes()).hexdigest()
    assert result.decision_sha256 == hashlib.sha256(decision.read_bytes()).hexdigest()
    assert result.operator == "reviewer"


def test_rejects_stale_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    manifest, readiness, decision = _package(tmp_path, report)
    readiness.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)

    with pytest.raises(ValueError, match="does not match"):
        module.verify_promotion_package(manifest, readiness, decision)


def test_rejects_non_approve_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    manifest, readiness, decision = _package(tmp_path, report)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["decision"] = "reject"
    decision.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)

    with pytest.raises(ValueError, match="approve decision"):
        module.verify_promotion_package(manifest, readiness, decision)


def test_rejects_digest_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    manifest, readiness, decision = _package(tmp_path, report)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    decision.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)

    with pytest.raises(ValueError, match="manifest digest"):
        module.verify_promotion_package(manifest, readiness, decision)


def test_rejects_not_ready_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report(ready=False)
    manifest, readiness, decision = _package(tmp_path, report)
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)

    with pytest.raises(ValueError, match="not ready"):
        module.verify_promotion_package(manifest, readiness, decision)


def test_rejects_manifest_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report()
    manifest, readiness, decision = _package(tmp_path, report)

    def mutate(_: Path) -> SimpleNamespace:
        manifest.write_text('{"status":"changed"}', encoding="utf-8")
        return report

    monkeypatch.setattr(module, "inspect_evidence_readiness", mutate)
    with pytest.raises(ValueError, match="changed during package verification"):
        module.verify_promotion_package(manifest, readiness, decision)
