from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from bharat.data.manifest import DatasetManifest

_APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected", "revoked"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class DatasetApproval:
    approval_id: str
    dataset_id: str
    manifest_digest: str
    approver: str
    approval_status: str
    approved_at: str
    license_reviewed: bool
    pii_reviewed: bool
    contamination_reviewed: bool
    safety_reviewed: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if self.approval_status not in _APPROVAL_STATUSES:
            raise ValueError(
                f"approval_status must be one of {sorted(_APPROVAL_STATUSES)}, "
                f"got {self.approval_status!r}"
            )
        if not _ISO_UTC_RE.match(self.approved_at):
            raise ValueError(
                f"approved_at must be ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ), "
                f"got {self.approved_at!r}"
            )
        if not _SHA256_RE.match(self.manifest_digest):
            raise ValueError(
                f"manifest_digest must be a 64-char lowercase hex string, "
                f"got {self.manifest_digest!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "dataset_id": self.dataset_id,
            "manifest_digest": self.manifest_digest,
            "approver": self.approver,
            "approval_status": self.approval_status,
            "approved_at": self.approved_at,
            "license_reviewed": self.license_reviewed,
            "pii_reviewed": self.pii_reviewed,
            "contamination_reviewed": self.contamination_reviewed,
            "safety_reviewed": self.safety_reviewed,
            "notes": self.notes,
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_approval_for_manifest(
    approval: DatasetApproval,
    manifest: DatasetManifest,
) -> list[str]:
    issues: list[str] = []

    if approval.dataset_id != manifest.dataset_id:
        issues.append(
            f"dataset_id mismatch: approval={approval.dataset_id!r} "
            f"vs manifest={manifest.dataset_id!r}"
        )

    expected_digest = manifest.digest()
    if approval.manifest_digest != expected_digest:
        issues.append(
            f"manifest_digest mismatch: approval={approval.manifest_digest!r} "
            f"vs computed={expected_digest!r}"
        )

    if approval.approval_status != "approved":
        issues.append(f"approval status must be 'approved', got {approval.approval_status!r}")

    if not approval.license_reviewed:
        issues.append("license_reviewed must be True")

    if not approval.pii_reviewed:
        issues.append("pii_reviewed must be True")

    if not approval.contamination_reviewed:
        issues.append("contamination_reviewed must be True")

    if not approval.safety_reviewed:
        issues.append("safety_reviewed must be True")

    return issues
