from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bharat.tokenizer.production_evidence_readiness import inspect_evidence_readiness

Decision = Literal["approve", "reject"]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PromotionDecisionRecord:
    manifest_sha256: str
    readiness_sha256: str
    decision: Decision
    operator: str
    rationale: str

    def to_canonical_dict(self) -> dict[str, str]:
        return {
            "schema_version": "tokenizer-promotion-decision-v1",
            "manifest_sha256": self.manifest_sha256,
            "readiness_sha256": self.readiness_sha256,
            "decision": self.decision,
            "operator": self.operator,
            "rationale": self.rationale,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_canonical_dict())


def build_promotion_decision(
    manifest_path: Path,
    readiness_path: Path,
    *,
    decision: Decision,
    operator: str,
    rationale: str,
) -> PromotionDecisionRecord:
    operator = operator.strip()
    rationale = rationale.strip()
    if not operator:
        raise ValueError("operator must be non-empty")
    if not rationale:
        raise ValueError("rationale must be non-empty")
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be 'approve' or 'reject'")

    manifest_sha256 = _sha256(manifest_path)
    readiness_bytes = readiness_path.read_bytes()
    report = inspect_evidence_readiness(manifest_path)
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("manifest changed during readiness validation")
    readiness = json.loads(readiness_bytes.decode("utf-8"))
    if readiness != report.to_canonical_dict():
        raise ValueError("readiness report does not match current manifest validation")
    if decision == "approve" and not report.ready_for_human_promotion:
        raise ValueError("cannot approve evidence that is not ready for human promotion")

    return PromotionDecisionRecord(
        manifest_sha256=manifest_sha256,
        readiness_sha256=_sha256_bytes(readiness_bytes),
        decision=decision,
        operator=operator,
        rationale=rationale,
    )


def write_promotion_decision(record: PromotionDecisionRecord, output_path: Path) -> str:
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    payload = record.canonical_bytes()
    owned = False
    try:
        with output.open("xb") as handle:
            owned = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if output.read_bytes() != payload:
            raise RuntimeError(f"byte-verification failed for {output}")
    except Exception:
        if owned:
            output.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()
