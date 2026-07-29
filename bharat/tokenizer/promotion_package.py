from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.production_evidence_readiness import inspect_evidence_readiness


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class PromotionPackageVerification:
    manifest_sha256: str
    readiness_sha256: str
    decision_sha256: str
    operator: str
    rationale: str


def verify_promotion_package(
    manifest_path: Path,
    readiness_path: Path,
    decision_path: Path,
) -> PromotionPackageVerification:
    """Verify a local approval package without mutating or promoting evidence."""

    manifest_sha256 = _sha256(manifest_path)
    readiness_bytes = readiness_path.read_bytes()
    decision_bytes = decision_path.read_bytes()

    report = inspect_evidence_readiness(manifest_path)
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("manifest changed during package verification")

    readiness = _load_json_object(readiness_bytes, label="readiness report")
    if readiness != report.to_canonical_dict():
        raise ValueError("readiness report does not match current manifest validation")
    if not report.ready_for_human_promotion:
        raise ValueError("candidate evidence is not ready for human promotion")

    decision = _load_json_object(decision_bytes, label="decision record")
    required = {
        "schema_version",
        "manifest_sha256",
        "readiness_sha256",
        "decision",
        "operator",
        "rationale",
    }
    if set(decision) != required:
        raise ValueError("decision record has unexpected or missing fields")
    if decision["schema_version"] != "tokenizer-promotion-decision-v1":
        raise ValueError("unsupported decision record schema")
    if decision["decision"] != "approve":
        raise ValueError("promotion package requires an approve decision")
    if decision["manifest_sha256"] != manifest_sha256:
        raise ValueError("decision record manifest digest does not match")

    readiness_sha256 = _sha256_bytes(readiness_bytes)
    if decision["readiness_sha256"] != readiness_sha256:
        raise ValueError("decision record readiness digest does not match")

    operator = decision["operator"]
    rationale = decision["rationale"]
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("decision operator must be non-empty")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("decision rationale must be non-empty")

    return PromotionPackageVerification(
        manifest_sha256=manifest_sha256,
        readiness_sha256=readiness_sha256,
        decision_sha256=_sha256_bytes(decision_bytes),
        operator=operator,
        rationale=rationale,
    )
