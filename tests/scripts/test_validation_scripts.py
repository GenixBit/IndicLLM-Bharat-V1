from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import ShardManifest, create_manifest
from scripts.validate_data_manifest import main as validate_manifest_main
from scripts.validate_dataset_approval import main as validate_approval_main
from scripts.validate_q8_0_compatibility import main as validate_q8_0_main


@pytest.fixture
def mock_manifest_and_approval(tmp_path: Path) -> tuple[Path, Path]:
    shard = ShardManifest(
        shard_id="shard-0000",
        index=0,
        record_start=0,
        record_end=10,
        bytes_utf8=1024,
        sha256="a" * 64,
        created_at="2026-01-01T00:00:00Z",
    )

    manifest = create_manifest(
        dataset_id="test_ds",
        source_id="sangraha",
        source_version="1.0.0",
        license="cc-by-4.0",
        language="hi",
        split="train",
        domain="general",
        shards=(shard,),
        records=10,
        bytes_utf8=1024,
        sha256="b" * 64,
        processing_config_digest="c" * 64,
        registry_digest="d" * 64,
        policy_digest="e" * 64,
        created_at="2026-01-01T00:00:00Z",
    )
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest.to_dict(), mf, indent=2)

    approval = DatasetApproval(
        approval_id="app_12345678",
        dataset_id="test_ds",
        manifest_digest=manifest.digest(),
        approver="lead_data_engineer",
        approval_status="approved",
        approved_at="2026-01-01T00:00:00Z",
        license_reviewed=True,
        pii_reviewed=True,
        contamination_reviewed=True,
        safety_reviewed=True,
        notes="Validated license and quality filters",
    )
    approval_path = tmp_path / "approval.json"
    with open(approval_path, "w", encoding="utf-8") as af:
        json.dump(approval.to_dict(), af, indent=2)

    return manifest_path, approval_path


class TestValidationScripts:
    def test_validate_data_manifest_valid(
        self, mock_manifest_and_approval: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest_path, _ = mock_manifest_and_approval
        ret = validate_manifest_main(["--manifest", str(manifest_path), "--json"])
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "valid"
        assert len(data["errors"]) == 0

    def test_validate_data_manifest_missing(self, tmp_path: Path) -> None:
        ret = validate_manifest_main(["--manifest", str(tmp_path / "missing.json")])
        assert ret == 1

    def test_validate_dataset_approval_valid(
        self, mock_manifest_and_approval: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest_path, approval_path = mock_manifest_and_approval
        ret = validate_approval_main(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--json",
            ]
        )
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "valid"
        assert len(data["errors"]) == 0

    def test_validate_dataset_approval_mismatch(
        self, mock_manifest_and_approval: tuple[Path, Path], tmp_path: Path
    ) -> None:
        manifest_path, _ = mock_manifest_and_approval
        mismatched_approval = DatasetApproval(
            approval_id="app_wrong",
            dataset_id="other_ds",
            manifest_digest=hashlib.sha256(b"wrong").hexdigest(),
            approver="lead_data_engineer",
            approval_status="approved",
            approved_at="2026-01-01T00:00:00Z",
            license_reviewed=True,
            pii_reviewed=True,
            contamination_reviewed=True,
            safety_reviewed=True,
        )
        bad_approval_path = tmp_path / "bad_approval.json"
        with open(bad_approval_path, "w", encoding="utf-8") as af:
            json.dump(mismatched_approval.to_dict(), af, indent=2)

        ret = validate_approval_main(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(bad_approval_path),
            ]
        )
        assert ret == 1

    @patch("subprocess.run")
    def test_validate_q8_0_compatibility_cli(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "All Q8_0 tests passed"
        mock_run.return_value.stderr = ""

        ret = validate_q8_0_main([])
        assert ret == 0
        mock_run.assert_called_once()
