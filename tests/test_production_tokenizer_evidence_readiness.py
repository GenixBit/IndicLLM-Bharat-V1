from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.tokenizer.production_evidence import ProductionEvidenceValidation
from bharat.tokenizer.production_evidence_readiness import (
    assess_evidence_readiness,
    write_readiness_report,
)


def _validation(
    *,
    status: str = "candidate",
    valid: bool = True,
    accepted: bool = False,
    errors: tuple[str, ...] = (),
) -> ProductionEvidenceValidation:
    return ProductionEvidenceValidation(
        manifest_sha256="a" * 64,
        status=status,
        valid=valid,
        accepted=accepted,
        errors=errors,
    )


def test_candidate_is_ready_for_human_promotion() -> None:
    report = assess_evidence_readiness(_validation())

    assert report.ready_for_human_promotion is True
    assert report.blockers == ()
    assert json.loads(report.canonical_bytes()) == report.to_canonical_dict()


def test_invalid_candidate_preserves_validation_errors() -> None:
    report = assess_evidence_readiness(
        _validation(valid=False, errors=("artifact digest mismatch",))
    )

    assert report.ready_for_human_promotion is False
    assert report.blockers == ("artifact digest mismatch",)


def test_accepted_manifest_is_not_a_promotion_candidate() -> None:
    report = assess_evidence_readiness(_validation(status="accepted", accepted=True))

    assert report.ready_for_human_promotion is False
    assert report.blockers == (
        "manifest status must be candidate before human promotion review",
        "accepted evidence does not require candidate promotion review",
    )


def test_writer_refuses_to_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_readiness.inspect_evidence_readiness",
        lambda _: assess_evidence_readiness(_validation()),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_readiness_report(tmp_path / "manifest.json", output)


def test_writer_is_canonical_and_returns_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        "bharat.tokenizer.production_evidence_readiness.inspect_evidence_readiness",
        lambda _: assess_evidence_readiness(_validation()),
    )

    digest = write_readiness_report(tmp_path / "manifest.json", output)

    assert len(digest) == 64
    assert output.read_bytes() == assess_evidence_readiness(_validation()).canonical_bytes()
