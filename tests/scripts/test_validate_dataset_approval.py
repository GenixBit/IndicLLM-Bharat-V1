from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import DatasetManifest


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.validate_dataset_approval", *args],
        capture_output=True,
        text=True,
    )


def _make_manifest(path: Path, dataset_id: str = "ds-test") -> DatasetManifest:
    sha = hashlib.sha256(b"content").hexdigest()
    manifest = DatasetManifest(
        manifest_version="1.0",
        dataset_id=dataset_id,
        source_id="src",
        source_version="1.0",
        created_at="2026-07-20T12:00:00Z",
        license="cc-by-4.0",
        language="en",
        split="train",
        records=100,
        bytes_utf8=50000,
        sha256=sha,
        processing_config_digest=sha,
        registry_digest=sha,
        policy_digest=sha,
    )
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return manifest


def _make_approval(
    path: Path,
    manifest: DatasetManifest | None = None,
    **overrides: object,
) -> None:
    kwargs: dict[str, object] = {
        "approval_id": "apr-001",
        "dataset_id": manifest.dataset_id if manifest else "ds-test",
        "manifest_digest": manifest.digest() if manifest else "a" * 64,
        "approver": "reviewer@example.com",
        "approval_status": "approved",
        "approved_at": "2026-07-20T12:00:00Z",
        "license_reviewed": True,
        "pii_reviewed": True,
        "contamination_reviewed": True,
        "safety_reviewed": True,
    }
    kwargs.update(overrides)
    approval = DatasetApproval(**kwargs)
    path.write_text(json.dumps(approval.to_dict(), indent=2))


class TestValidateDatasetApprovalCLI:
    def test_valid(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest)
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode == 0
        assert "Approval valid" in result.stdout

    def test_valid_json_output(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest)
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--json",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "valid"

    def test_failure_exits_nonzero(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, approval_status="pending")
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0
        assert result.stderr

    def test_json_failure_output(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, approval_status="pending")
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--json",
            ]
        )
        data = json.loads(result.stdout)
        assert data["status"] == "invalid"
        assert len(data["errors"]) > 0

    def test_missing_manifest_file(self, tmp_path: Path) -> None:
        approval_path = tmp_path / "approval.json"
        _make_approval(approval_path)
        result = run_cli(
            [
                "--manifest",
                str(tmp_path / "nonexistent.json"),
                "--approval",
                str(approval_path),
            ]
        )
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_missing_approval_file(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        _make_manifest(manifest_path)
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(tmp_path / "nonexistent.json"),
            ]
        )
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_approved_passes(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest)
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode == 0

    def test_pending_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, approval_status="pending")
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0

    def test_rejected_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, approval_status="rejected")
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0

    def test_revoked_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, approval_status="revoked")
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0

    def test_manifest_digest_mismatch_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        manifest = _make_manifest(manifest_path)
        _make_approval(approval_path, manifest=manifest, manifest_digest="b" * 64)
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0

    def test_dataset_id_mismatch_fails(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        approval_path = tmp_path / "approval.json"
        _make_manifest(manifest_path, dataset_id="ds-a")
        _make_approval(approval_path, dataset_id="ds-b")
        result = run_cli(["--manifest", str(manifest_path), "--approval", str(approval_path)])
        assert result.returncode != 0
