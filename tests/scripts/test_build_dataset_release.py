from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import DatasetManifest, ShardManifest


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.build_dataset_release", *args],
        capture_output=True,
        text=True,
    )


def _setup_build_test(tmp_path: Path) -> tuple[Path, Path, Path]:
    sha = hashlib.sha256(b"content").hexdigest()

    shard_content = b"shard data"
    shard_sha = hashlib.sha256(shard_content).hexdigest()

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_path = shards_dir / "shard-0000"
    shard_path.write_bytes(shard_content)

    shard_manifest = ShardManifest(
        shard_id="shard-0000",
        index=0,
        record_start=0,
        record_end=100,
        bytes_utf8=1000,
        sha256=shard_sha,
        created_at="2026-07-20T12:00:00Z",
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

    output_dir = tmp_path / "release"

    return manifest_path, approval_path, output_dir


class TestBuildDatasetReleaseCLI:
    def test_build_succeeds(self, tmp_path: Path) -> None:
        manifest_path, approval_path, output_dir = _setup_build_test(tmp_path)
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode == 0
        assert (output_dir / "dataset_release.json").exists()
        assert (output_dir / "audit_report.json").exists()

    def test_json_output(self, tmp_path: Path) -> None:
        manifest_path, approval_path, output_dir = _setup_build_test(tmp_path)
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "success"

    def test_failure_exits_nonzero(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "nonexistent.json"
        approval_path = tmp_path / "approval.json"
        approval_path.write_text("{}")
        output_dir = tmp_path / "release"
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0

    def test_tampered_shard_fails(self, tmp_path: Path) -> None:
        manifest_path, approval_path, output_dir = _setup_build_test(tmp_path)
        shard_path = tmp_path / "shards" / "shard-0000"
        shard_path.write_bytes(b"tampered data")
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0
        assert "Tampered shard" in result.stderr or "error:" in result.stderr

    def test_missing_shard_fails(self, tmp_path: Path) -> None:
        manifest_path, approval_path, output_dir = _setup_build_test(tmp_path)
        shard_path = tmp_path / "shards" / "shard-0000"
        shard_path.unlink()
        result = run_cli(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0
