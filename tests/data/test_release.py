from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import DatasetManifest, ShardManifest
from bharat.data.release import (
    DatasetAuditReport,
    DatasetRelease,
    DatasetReleaseBuilder,
)


def _digest(data: bytes = b"test") -> str:
    return hashlib.sha256(data).hexdigest()


def _make_shard_manifest(
    index: int = 0,
    record_start: int = 0,
    record_end: int = 100,
    sha256: str = "",
) -> ShardManifest:
    return ShardManifest(
        shard_id=f"shard-{index:04d}",
        index=index,
        record_start=record_start,
        record_end=record_end,
        bytes_utf8=1000,
        sha256=sha256,
        created_at="2026-07-20T12:00:00Z",
    )


class TestDatasetRelease:
    def test_minimal_valid(self) -> None:
        sha = _digest()
        rel = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        assert rel.release_id == "rel-001"
        assert rel.digest()

    def test_to_dict_roundtrip(self) -> None:
        sha = _digest()
        r1 = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        d = r1.to_dict()
        r2 = DatasetRelease(
            release_id=d["release_id"],
            dataset_id=d["dataset_id"],
            manifest_digest=d["manifest_digest"],
            approval_digest=d["approval_digest"],
            shard_count=d["shard_count"],
            records=d["records"],
            bytes_utf8=d["bytes_utf8"],
            created_at=d["created_at"],
            package_sha256=d["package_sha256"],
        )
        assert r1 == r2

    def test_digest_deterministic(self) -> None:
        sha = _digest()
        r1 = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        r2 = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        assert r1.digest() == r2.digest()

    def test_digest_changes_with_field(self) -> None:
        sha = _digest()
        r1 = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        r2 = DatasetRelease(
            release_id="rel-002",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        assert r1.digest() != r2.digest()

    def test_invalid_release_id_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="release_id"):
            DatasetRelease(
                release_id="",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=1,
                records=100,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_invalid_manifest_digest_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="manifest_digest"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest="short",
                approval_digest=sha,
                shard_count=1,
                records=100,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_invalid_approval_digest_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="approval_digest"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest="short",
                shard_count=1,
                records=100,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_negative_shard_count_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="shard_count"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=-1,
                records=100,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_negative_records_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="records"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=1,
                records=-1,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_negative_bytes_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="bytes_utf8"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=1,
                records=100,
                bytes_utf8=-1,
                created_at="2026-07-20T12:00:00Z",
                package_sha256=sha,
            )

    def test_invalid_created_at_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="created_at"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=1,
                records=100,
                bytes_utf8=1000,
                created_at="bad-ts",
                package_sha256=sha,
            )

    def test_invalid_package_sha256_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="package_sha256"):
            DatasetRelease(
                release_id="rel-001",
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_count=1,
                records=100,
                bytes_utf8=1000,
                created_at="2026-07-20T12:00:00Z",
                package_sha256="short",
            )

    def test_release_json_deterministic(self) -> None:
        sha = _digest()
        r1 = DatasetRelease(
            release_id="rel-001",
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_count=1,
            records=100,
            bytes_utf8=1000,
            created_at="2026-07-20T12:00:00Z",
            package_sha256=sha,
        )
        json1 = json.dumps(r1.to_dict(), sort_keys=True)
        json2 = json.dumps(r1.to_dict(), sort_keys=True)
        assert json1 == json2


class TestDatasetAuditReport:
    def test_minimal_valid(self) -> None:
        sha = _digest()
        r = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        assert r.dataset_id == "ds-test"
        assert r.digest()

    def test_to_dict_roundtrip(self) -> None:
        sha = _digest()
        r1 = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        d = r1.to_dict()
        r2 = DatasetAuditReport(
            dataset_id=d["dataset_id"],
            manifest_digest=d["manifest_digest"],
            approval_digest=d["approval_digest"],
            shard_checks_passed=d["shard_checks_passed"],
            approval_checks_passed=d["approval_checks_passed"],
            total_records=d["total_records"],
            total_bytes_utf8=d["total_bytes_utf8"],
            issues=tuple(d["issues"]),
        )
        assert r1 == r2

    def test_digest_deterministic(self) -> None:
        sha = _digest()
        r1 = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        r2 = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        assert r1.digest() == r2.digest()

    def test_digest_changes_with_field(self) -> None:
        sha = _digest()
        r1 = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        r2 = DatasetAuditReport(
            dataset_id="ds-test-other",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        assert r1.digest() != r2.digest()

    def test_audit_report_deterministic(self) -> None:
        sha = _digest()
        r = DatasetAuditReport(
            dataset_id="ds-test",
            manifest_digest=sha,
            approval_digest=sha,
            shard_checks_passed=True,
            approval_checks_passed=True,
            total_records=100,
            total_bytes_utf8=1000,
        )
        json1 = json.dumps(r.to_dict(), sort_keys=True)
        json2 = json.dumps(r.to_dict(), sort_keys=True)
        assert json1 == json2

    def test_negative_total_records_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="total_records"):
            DatasetAuditReport(
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_checks_passed=True,
                approval_checks_passed=True,
                total_records=-1,
                total_bytes_utf8=1000,
            )

    def test_negative_total_bytes_raises(self) -> None:
        sha = _digest()
        with pytest.raises(ValueError, match="total_bytes_utf8"):
            DatasetAuditReport(
                dataset_id="ds-test",
                manifest_digest=sha,
                approval_digest=sha,
                shard_checks_passed=True,
                approval_checks_passed=True,
                total_records=100,
                total_bytes_utf8=-1,
            )


def _setup_builder_test(tmp_path: Path) -> tuple[Path, Path, DatasetManifest, DatasetApproval]:
    sha = _digest()

    shard_content = b"shard data content"
    shard_sha = hashlib.sha256(shard_content).hexdigest()

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_path = shards_dir / "shard-0000"
    shard_path.write_bytes(shard_content)

    shard_manifest = _make_shard_manifest(
        index=0,
        record_start=0,
        record_end=100,
        sha256=shard_sha,
    )

    manifest = DatasetManifest(
        manifest_version="1.0",
        dataset_id="ds-test",
        source_id="src",
        source_version="1.0",
        created_at="2026-07-20T12:00:00Z",
        license="cc-by-4.0",
        language="en",
        split="train",
        records=100,
        bytes_utf8=1000,
        sha256=sha,
        processing_config_digest=sha,
        registry_digest=sha,
        policy_digest=sha,
        shards=(shard_manifest,),
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    approval = DatasetApproval(
        approval_id="apr-001",
        dataset_id="ds-test",
        manifest_digest=manifest.digest(),
        approver="reviewer@example.com",
        approval_status="approved",
        approved_at="2026-07-20T12:00:00Z",
        license_reviewed=True,
        pii_reviewed=True,
        contamination_reviewed=True,
        safety_reviewed=True,
    )
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval.to_dict(), indent=2))

    return manifest_path, approval_path, manifest, approval


class TestDatasetReleaseBuilder:
    def test_build_succeeds(self, tmp_path: Path) -> None:
        manifest_path, approval_path, manifest, approval = _setup_builder_test(tmp_path)
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        release, audit = builder.build(manifest_path, approval_path, output_dir)
        assert release.dataset_id == manifest.dataset_id
        assert release.manifest_digest == manifest.digest()
        assert release.approval_digest == approval.digest()
        assert release.shard_count == 1
        assert release.records == 100
        assert audit.shard_checks_passed is True
        assert audit.approval_checks_passed is True
        assert (output_dir / "dataset_release.json").exists()
        assert (output_dir / "audit_report.json").exists()

    def test_release_json_deterministic(self, tmp_path: Path) -> None:
        manifest_path, approval_path, _, _ = _setup_builder_test(tmp_path)
        output_dir = tmp_path / "release1"
        builder = DatasetReleaseBuilder()
        release1, _ = builder.build(manifest_path, approval_path, output_dir)

        output_dir2 = tmp_path / "release2"
        release2, _ = builder.build(manifest_path, approval_path, output_dir2)

        assert release1.to_dict() == release2.to_dict()

    def test_audit_report_deterministic(self, tmp_path: Path) -> None:
        manifest_path, approval_path, _, _ = _setup_builder_test(tmp_path)
        output_dir = tmp_path / "release1"
        builder = DatasetReleaseBuilder()
        _, audit1 = builder.build(manifest_path, approval_path, output_dir)

        output_dir2 = tmp_path / "release2"
        _, audit2 = builder.build(manifest_path, approval_path, output_dir2)

        assert audit1.to_dict() == audit2.to_dict()

    def test_tampered_shard_fails(self, tmp_path: Path) -> None:
        manifest_path, approval_path, _, _ = _setup_builder_test(tmp_path)
        shard_path = tmp_path / "shards" / "shard-0000"
        shard_path.write_bytes(b"tampered content")
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(ValueError, match="Tampered shard"):
            builder.build(manifest_path, approval_path, output_dir)

    def test_missing_shard_fails(self, tmp_path: Path) -> None:
        manifest_path, approval_path, _, _ = _setup_builder_test(tmp_path)
        shard_path = tmp_path / "shards" / "shard-0000"
        shard_path.unlink()
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(ValueError, match="Missing shard"):
            builder.build(manifest_path, approval_path, output_dir)

    def test_remote_manifest_path_rejected(self, tmp_path: Path) -> None:
        approval_path = tmp_path / "approval.json"
        approval_path.write_text("{}")
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(ValueError, match="Remote manifest path"):
            builder.build("s3://bucket/manifest.json", approval_path, output_dir)

    def test_remote_approval_path_rejected(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{}")
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(ValueError, match="Remote approval path"):
            builder.build(manifest_path, "gs://bucket/approval.json", output_dir)

    def test_missing_manifest_file_fails(self, tmp_path: Path) -> None:
        approval_path = tmp_path / "approval.json"
        approval_path.write_text("{}")
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            builder.build(tmp_path / "nonexistent.json", approval_path, output_dir)

    def test_missing_approval_file_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{}")
        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(FileNotFoundError, match="Approval not found"):
            builder.build(manifest_path, tmp_path / "nonexistent.json", output_dir)

    def test_pending_approval_fails_build(self, tmp_path: Path) -> None:
        sha = _digest()
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        shard_path = shards_dir / "shard-0000"
        shard_path.write_bytes(b"data")
        shard_sha = hashlib.sha256(b"data").hexdigest()

        manifest = DatasetManifest(
            manifest_version="1.0",
            dataset_id="ds-test",
            source_id="src",
            source_version="1.0",
            created_at="2026-07-20T12:00:00Z",
            license="cc-by-4.0",
            language="en",
            split="train",
            records=100,
            bytes_utf8=1000,
            sha256=sha,
            processing_config_digest=sha,
            registry_digest=sha,
            policy_digest=sha,
            shards=(_make_shard_manifest(sha256=shard_sha),),
        )
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        approval = DatasetApproval(
            approval_id="apr-001",
            dataset_id="ds-test",
            manifest_digest=manifest.digest(),
            approver="reviewer",
            approval_status="pending",
            approved_at="2026-07-20T12:00:00Z",
            license_reviewed=True,
            pii_reviewed=True,
            contamination_reviewed=True,
            safety_reviewed=True,
        )
        approval_path = tmp_path / "approval.json"
        approval_path.write_text(json.dumps(approval.to_dict(), indent=2))

        output_dir = tmp_path / "release"
        builder = DatasetReleaseBuilder()
        with pytest.raises(ValueError, match="Approval validation failed"):
            builder.build(manifest_path, approval_path, output_dir)
