from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.production_evidence import (
    ProductionEvidenceValidation,
    validate_production_evidence,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class EvidenceReadinessReport:
    manifest_sha256: str
    manifest_status: str
    structurally_valid: bool
    accepted: bool
    ready_for_human_promotion: bool
    blockers: tuple[str, ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tokenizer-production-evidence-readiness-v1",
            "manifest_sha256": self.manifest_sha256,
            "manifest_status": self.manifest_status,
            "structurally_valid": self.structurally_valid,
            "accepted": self.accepted,
            "ready_for_human_promotion": self.ready_for_human_promotion,
            "blockers": list(self.blockers),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_canonical_dict())


def assess_evidence_readiness(
    validation: ProductionEvidenceValidation,
) -> EvidenceReadinessReport:
    blockers = list(validation.errors)
    if validation.valid and validation.status != "candidate":
        blockers.append("manifest status must be candidate before human promotion review")
    if validation.valid and validation.accepted:
        blockers.append("accepted evidence does not require candidate promotion review")
    ready = validation.valid and validation.status == "candidate" and not validation.accepted
    return EvidenceReadinessReport(
        manifest_sha256=validation.manifest_sha256,
        manifest_status=validation.status,
        structurally_valid=validation.valid,
        accepted=validation.accepted,
        ready_for_human_promotion=ready,
        blockers=tuple(blockers),
    )


def inspect_evidence_readiness(manifest_path: Path) -> EvidenceReadinessReport:
    return assess_evidence_readiness(validate_production_evidence(manifest_path))


def write_readiness_report(manifest_path: Path, output_path: Path) -> str:
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    report = inspect_evidence_readiness(manifest_path)
    payload = report.canonical_bytes()
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if output.read_bytes() != payload:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"byte-verification failed for {output}")
    return hashlib.sha256(payload).hexdigest()
