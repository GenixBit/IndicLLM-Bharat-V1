from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_decision as module


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    readiness = tmp_path / "readiness.json"
    manifest.write_text('{"status":"candidate"}', encoding="utf-8")
    readiness.write_text(
        json.dumps(
            {
                "schema_version": "tokenizer-production-evidence-readiness-v1",
                "manifest_sha256": "abc",
                "manifest_status": "candidate",
                "structurally_valid": True,
                "accepted": False,
                "ready_for_human_promotion": True,
                "blockers": [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest, readiness


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


def test_build_and_write_canonical_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, readiness = _paths(tmp_path)
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: _report())
    record = module.build_promotion_decision(
        manifest,
        readiness,
        decision="approve",
        operator=" reviewer ",
        rationale=" verified offline evidence ",
    )
    output = tmp_path / "decision.json"
    digest = module.write_promotion_decision(record, output)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_text()) == record.to_canonical_dict()
    assert record.operator == "reviewer"


def test_approve_requires_ready_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, readiness = _paths(tmp_path)
    report = _report(ready=False)
    readiness.write_text(json.dumps(report.to_canonical_dict()), encoding="utf-8")
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)
    with pytest.raises(ValueError, match="not ready"):
        module.build_promotion_decision(
            manifest,
            readiness,
            decision="approve",
            operator="reviewer",
            rationale="approve",
        )


def test_reject_allowed_when_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, readiness = _paths(tmp_path)
    report = _report(ready=False)
    readiness.write_text(json.dumps(report.to_canonical_dict()), encoding="utf-8")
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: report)
    record = module.build_promotion_decision(
        manifest,
        readiness,
        decision="reject",
        operator="reviewer",
        rationale="blockers remain",
    )
    assert record.decision == "reject"


def test_rejects_stale_readiness_and_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, readiness = _paths(tmp_path)
    monkeypatch.setattr(module, "inspect_evidence_readiness", lambda _: _report())
    readiness.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        module.build_promotion_decision(
            manifest,
            readiness,
            decision="reject",
            operator="reviewer",
            rationale="stale",
        )

    record = module.PromotionDecisionRecord("a", "b", "reject", "reviewer", "reason")
    output = tmp_path / "decision.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.write_promotion_decision(record, output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_create_race_does_not_delete_unowned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = module.PromotionDecisionRecord("a", "b", "reject", "reviewer", "reason")
    output = (tmp_path / "decision.json").resolve()
    output.write_text("raced", encoding="utf-8")
    original_exists = Path.exists

    def report_missing(path: Path) -> bool:
        if path == output:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", report_missing)
    with pytest.raises(FileExistsError):
        module.write_promotion_decision(record, output)
    assert output.read_text(encoding="utf-8") == "raced"


def test_rejects_manifest_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, readiness = _paths(tmp_path)

    def mutate_manifest(_: Path) -> SimpleNamespace:
        manifest.write_text('{"status":"replaced"}', encoding="utf-8")
        return _report()

    monkeypatch.setattr(module, "inspect_evidence_readiness", mutate_manifest)
    with pytest.raises(ValueError, match="changed during readiness validation"):
        module.build_promotion_decision(
            manifest,
            readiness,
            decision="approve",
            operator="reviewer",
            rationale="verified",
        )
