from __future__ import annotations

import hashlib

import pytest

from bharat.data.approval import DatasetApproval, validate_approval_for_manifest
from bharat.data.manifest import DatasetManifest


def _make_approval(**overrides: object) -> DatasetApproval:
    kwargs: dict[str, object] = {
        "approval_id": "apr-001",
        "dataset_id": "ds-test",
        "manifest_digest": "a" * 64,
        "approver": "reviewer@example.com",
        "approval_status": "approved",
        "approved_at": "2026-07-20T12:00:00Z",
        "license_reviewed": True,
        "pii_reviewed": True,
        "contamination_reviewed": True,
        "safety_reviewed": True,
    }
    kwargs.update(overrides)
    return DatasetApproval(**kwargs)


def _make_manifest(dataset_id: str = "ds-test") -> DatasetManifest:
    sha = hashlib.sha256(b"dummy").hexdigest()
    return DatasetManifest(
        manifest_version="1.0",
        dataset_id=dataset_id,
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
    )


class TestDatasetApproval:
    def test_minimal_valid(self) -> None:
        a = _make_approval()
        assert a.approval_id == "apr-001"
        assert a.digest()

    def test_to_dict_roundtrip(self) -> None:
        a1 = _make_approval()
        d = a1.to_dict()
        a2 = DatasetApproval(**d)
        assert a1 == a2

    def test_digest_deterministic(self) -> None:
        a1 = _make_approval()
        a2 = _make_approval()
        assert a1.digest() == a2.digest()

    def test_digest_changes_with_field(self) -> None:
        a1 = _make_approval()
        a2 = _make_approval(notes="different")
        assert a1.digest() != a2.digest()

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="approval_status"):
            _make_approval(approval_status="invalid")

    def test_invalid_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="approved_at"):
            _make_approval(approved_at="bad-ts")

    def test_invalid_manifest_digest_raises(self) -> None:
        with pytest.raises(ValueError, match="manifest_digest"):
            _make_approval(manifest_digest="short")


class TestValidateApprovalForManifest:
    def test_approved_passes(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert issues == []

    def test_pending_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_status="pending",
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("approved" in i for i in issues)

    def test_rejected_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_status="rejected",
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("approved" in i for i in issues)

    def test_revoked_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_status="revoked",
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("approved" in i for i in issues)

    def test_manifest_digest_mismatch_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest="b" * 64,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("manifest_digest" in i for i in issues)

    def test_dataset_id_mismatch_fails(self) -> None:
        manifest = _make_manifest(dataset_id="ds-a")
        approval = _make_approval(
            dataset_id="ds-b",
            manifest_digest=manifest.digest(),
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("dataset_id" in i for i in issues)

    def test_missing_license_review_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            license_reviewed=False,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("license_reviewed" in i for i in issues)

    def test_missing_pii_review_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            pii_reviewed=False,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("pii_reviewed" in i for i in issues)

    def test_missing_contamination_review_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            contamination_reviewed=False,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("contamination_reviewed" in i for i in issues)

    def test_missing_safety_review_fails(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            safety_reviewed=False,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        assert any("safety_reviewed" in i for i in issues)

    def test_pending_allows_missing_reviews(self) -> None:
        manifest = _make_manifest()
        approval = _make_approval(
            dataset_id=manifest.dataset_id,
            manifest_digest=manifest.digest(),
            approval_status="pending",
            license_reviewed=False,
            pii_reviewed=False,
            contamination_reviewed=False,
            safety_reviewed=False,
        )
        issues = validate_approval_for_manifest(approval, manifest)
        status_issues = [i for i in issues if "approved" in i]
        review_issues = [i for i in issues if "reviewed" in i]
        assert len(status_issues) >= 1
        assert len(review_issues) == 4
