from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import ShardManifest, create_manifest
from scripts.build_dataset_release import main as build_release_main
from scripts.compute_data_stats import main as compute_stats_main


class TestDataScripts:
    def test_compute_data_stats_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data_file = tmp_path / "sample.txt"
        data_file.write_text(
            "नमस्ते भारत। यह एक बहुत अच्छा और विस्तृत परीक्षण दस्तावेज है। " * 5, encoding="utf-8"
        )

        ret = compute_stats_main(["--input", str(data_file), "--json"])
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["record_count"] == 1
        assert data["total_chars"] > 0

    def test_compute_data_stats_jsonl(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        lines = [
            "भारत एक महान और समृद्ध देश है।",
            "This is a multilingual test string.",
        ]
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        ret = compute_stats_main(["--input", str(jsonl_file)])
        assert ret == 0

        captured = capsys.readouterr()
        assert "Dataset Statistics" in captured.out
        assert "Records:              2" in captured.out

    def test_build_dataset_release_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        shard_content = b"\x00\x01\x02\x03" * 256
        shard_sha = hashlib.sha256(shard_content).hexdigest()
        shard_file = shards_dir / "shard-0000"
        shard_file.write_bytes(shard_content)

        shard = ShardManifest(
            shard_id="shard-0000",
            index=0,
            record_start=0,
            record_end=10,
            bytes_utf8=len(shard_content),
            sha256=shard_sha,
            created_at="2026-01-01T00:00:00Z",
        )

        manifest = create_manifest(
            dataset_id="test_hi_dataset",
            source_id="sangraha",
            source_version="1.0.0",
            license="cc-by-4.0",
            language="hi",
            split="train",
            domain="general",
            shards=(shard,),
            records=10,
            bytes_utf8=len(shard_content),
            sha256=shard_sha,
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
            dataset_id="test_hi_dataset",
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

        out_dir = tmp_path / "release_out"
        ret = build_release_main(
            [
                "--manifest",
                str(manifest_path),
                "--approval",
                str(approval_path),
                "--output-dir",
                str(out_dir),
                "--json",
            ]
        )
        assert ret == 0

        captured = capsys.readouterr()
        res = json.loads(captured.out)
        assert res["status"] == "success"
        assert res["dataset_id"] == "test_hi_dataset"
        assert (out_dir / "dataset_release.json").is_file()
