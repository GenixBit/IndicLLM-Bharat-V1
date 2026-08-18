from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.data.manifest import ShardManifest, create_manifest
from scripts.plan_data_shards import main as plan_shards_main
from scripts.prepare_local_data import main as prepare_local_main
from scripts.run_serving_control_smoke import main as serving_smoke_main


class TestPipelineScripts:
    def test_plan_data_shards_cli(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        shard = ShardManifest(
            shard_id="shard-0000",
            index=0,
            record_start=0,
            record_end=5000,
            bytes_utf8=100000,
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
            records=5000,
            bytes_utf8=100000,
            sha256="b" * 64,
            processing_config_digest="c" * 64,
            registry_digest="d" * 64,
            policy_digest="e" * 64,
            created_at="2026-01-01T00:00:00Z",
        )
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest.to_dict(), mf)

        ret = plan_shards_main(
            [
                "--manifest",
                str(manifest_path),
                "--max-records",
                "1000",
                "--json",
            ]
        )
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["shard_count"] == 5
        assert data["total_records"] == 5000

    def test_prepare_local_data_cli_dry_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        raw_file = tmp_path / "input.jsonl"
        records = [
            json.dumps({"text": "नमस्ते भारत। यह एक बहुत अच्छा परीक्षण है।" * 3}),
            json.dumps({"text": "भारतवर्ष एक समृद्ध और सुंदर राष्ट्र है।" * 3}),
        ]
        raw_file.write_text("\n".join(records), encoding="utf-8")

        out_dir = tmp_path / "output_governed"
        ret = prepare_local_main(
            [
                "--input",
                str(raw_file),
                "--source-id",
                "test_source",
                "--source-version",
                "1.0.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "hi",
                "--output-dir",
                str(out_dir),
                "--dry-run",
                "--json",
            ]
        )
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["dry_run"] is True
        assert data["total_records"] == 2

    def test_run_serving_control_smoke_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_events = tmp_path / "events.jsonl"
        ret = serving_smoke_main(
            [
                "--prompt",
                "नमस्ते भारत",
                "--output",
                str(out_events),
                "--json",
            ]
        )
        assert ret == 0
        assert out_events.is_file()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "success"
        assert data["event_count"] > 0
        assert "metrics" in data
