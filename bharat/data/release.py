from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bharat.data.approval import DatasetApproval, validate_approval_for_manifest
from bharat.data.manifest import DatasetManifest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetRelease:
    release_id: str
    dataset_id: str
    manifest_digest: str
    approval_digest: str
    shard_count: int
    records: int
    bytes_utf8: int
    created_at: str
    package_sha256: str

    def __post_init__(self) -> None:
        if not self.release_id:
            raise ValueError("release_id must be a non-empty string")
        if not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not _SHA256_RE.match(self.manifest_digest):
            raise ValueError(
                f"manifest_digest must be a 64-char lowercase hex string, "
                f"got {self.manifest_digest!r}"
            )
        if not _SHA256_RE.match(self.approval_digest):
            raise ValueError(
                f"approval_digest must be a 64-char lowercase hex string, "
                f"got {self.approval_digest!r}"
            )
        if self.shard_count < 0:
            raise ValueError(f"shard_count must be non-negative, got {self.shard_count}")
        if self.records < 0:
            raise ValueError(f"records must be non-negative, got {self.records}")
        if self.bytes_utf8 < 0:
            raise ValueError(f"bytes_utf8 must be non-negative, got {self.bytes_utf8}")
        if not _ISO_UTC_RE.match(self.created_at):
            raise ValueError(
                f"created_at must be ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ), "
                f"got {self.created_at!r}"
            )
        if not _SHA256_RE.match(self.package_sha256):
            raise ValueError(
                f"package_sha256 must be a 64-char lowercase hex string, "
                f"got {self.package_sha256!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "dataset_id": self.dataset_id,
            "manifest_digest": self.manifest_digest,
            "approval_digest": self.approval_digest,
            "shard_count": self.shard_count,
            "records": self.records,
            "bytes_utf8": self.bytes_utf8,
            "created_at": self.created_at,
            "package_sha256": self.package_sha256,
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetAuditReport:
    dataset_id: str
    manifest_digest: str
    approval_digest: str
    shard_checks_passed: bool
    approval_checks_passed: bool
    total_records: int
    total_bytes_utf8: int
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if self.total_records < 0:
            raise ValueError(f"total_records must be non-negative, got {self.total_records}")
        if self.total_bytes_utf8 < 0:
            raise ValueError(f"total_bytes_utf8 must be non-negative, got {self.total_bytes_utf8}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "manifest_digest": self.manifest_digest,
            "approval_digest": self.approval_digest,
            "shard_checks_passed": self.shard_checks_passed,
            "approval_checks_passed": self.approval_checks_passed,
            "total_records": self.total_records,
            "total_bytes_utf8": self.total_bytes_utf8,
            "issues": list(self.issues),
        }
        return d

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DatasetReleaseBuilder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _is_remote_url(path: str | Path) -> bool:
        return bool(_URL_RE.match(str(path)))

    def build(
        self,
        manifest_path: str | Path,
        approval_path: str | Path,
        output_dir: str | Path,
    ) -> tuple[DatasetRelease, DatasetAuditReport]:
        manifest_str = str(manifest_path)
        approval_str = str(approval_path)

        if self._is_remote_url(manifest_str):
            raise ValueError(f"Remote manifest path rejected: {manifest_str}")
        if self._is_remote_url(approval_str):
            raise ValueError(f"Remote approval path rejected: {approval_str}")

        manifest_path = Path(manifest_str)
        approval_path = Path(approval_str)
        output_dir = Path(output_dir)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not approval_path.exists():
            raise FileNotFoundError(f"Approval not found: {approval_path}")

        raw_manifest = manifest_path.read_text(encoding="utf-8")
        manifest_data = json.loads(raw_manifest)
        manifest = DatasetManifest.from_dict(manifest_data)

        raw_approval = approval_path.read_text(encoding="utf-8")
        approval_data = json.loads(raw_approval)
        approval = DatasetApproval(**approval_data)

        approval_issues = validate_approval_for_manifest(approval, manifest)
        if approval_issues:
            raise ValueError(f"Approval validation failed: {'; '.join(approval_issues)}")

        manifest_parent = manifest_path.parent
        shards_dir = manifest_parent / "shards"

        shard_errors: list[str] = []
        total_shard_records = 0
        total_shard_bytes = 0
        combined_sha = hashlib.sha256()

        sorted_shards = sorted(manifest.shards, key=lambda s: s.index)
        for i, shard in enumerate(sorted_shards):
            if shard.index != i:
                shard_errors.append(f"Shard index gap: expected {i}, got {shard.index}")

            expected_shard_path = shards_dir / shard.shard_id
            if not expected_shard_path.exists():
                shard_errors.append(f"Missing shard file: {expected_shard_path}")
                continue

            if shard.sha256:
                computed = hashlib.sha256(expected_shard_path.read_bytes()).hexdigest()
                if computed != shard.sha256:
                    shard_errors.append(
                        f"Tampered shard {shard.shard_id}: declared "
                        f"sha256={shard.sha256}, computed={computed}"
                    )

            combined_sha.update(expected_shard_path.read_bytes())
            total_shard_records += shard.record_end - shard.record_start
            total_shard_bytes += shard.bytes_utf8

        for i in range(1, len(sorted_shards)):
            prev = sorted_shards[i - 1]
            curr = sorted_shards[i]
            if curr.record_start != prev.record_end:
                shard_errors.append(
                    f"Non-contiguous shards: shard {prev.index} ends at "
                    f"{prev.record_end} but shard {curr.index} starts at "
                    f"{curr.record_start}"
                )

        if shard_errors:
            raise ValueError(f"Shard validation failed: {'; '.join(shard_errors)}")

        package_sha256 = combined_sha.hexdigest()
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        shard_count = len(manifest.shards)

        release = DatasetRelease(
            release_id=f"release-{manifest.dataset_id}-{now[:10]}",
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_digest=approval.digest(),
            shard_count=shard_count,
            records=total_shard_records,
            bytes_utf8=total_shard_bytes,
            created_at=now,
            package_sha256=package_sha256,
        )

        audit_report = DatasetAuditReport(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_digest=approval.digest(),
            shard_checks_passed=len(shard_errors) == 0,
            approval_checks_passed=len(approval_issues) == 0,
            total_records=total_shard_records,
            total_bytes_utf8=total_shard_bytes,
            issues=(),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        release_path = output_dir / "dataset_release.json"
        release_path.write_text(json.dumps(release.to_dict(), indent=2), encoding="utf-8")
        audit_path = output_dir / "audit_report.json"
        audit_path.write_text(json.dumps(audit_report.to_dict(), indent=2), encoding="utf-8")

        return release, audit_report
