from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bharat.data.manifest import create_manifest


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.plan_data_shards", *args],
        capture_output=True,
        text=True,
    )


_SAMPLE_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_SAMPLE_DIGEST = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"


def _make_manifest(path: Path, records: int = 1000, bytes_utf8: int = 500000) -> Path:
    m = create_manifest(
        dataset_id="test_ds",
        source_id="test_src",
        source_version="1.0.0",
        license="cc-by-4.0",
        language="en",
        split="train",
        records=records,
        bytes_utf8=bytes_utf8,
        sha256=_SAMPLE_SHA256,
        processing_config_digest=_SAMPLE_DIGEST,
        registry_digest=_SAMPLE_DIGEST,
        policy_digest=_SAMPLE_DIGEST,
    )
    path.write_text(json.dumps(m.to_dict(), indent=2))
    return path


class TestPlanShardsCLI:
    def test_single_shard(self, tmp_path):
        manifest_path = _make_manifest(tmp_path / "manifest.json", records=100)
        result = run_cli(["--manifest", str(manifest_path)])
        assert result.returncode == 0
        assert "Shard plan" in result.stdout
        assert "Shards:" in result.stdout

    def test_multiple_shards(self, tmp_path):
        manifest_path = _make_manifest(tmp_path / "manifest.json", records=25000)
        result = run_cli(["--manifest", str(manifest_path), "--max-records", "10000"])
        assert result.returncode == 0
        assert "Shards:        3" in result.stdout

    def test_json_output(self, tmp_path):
        manifest_path = _make_manifest(tmp_path / "manifest.json", records=100)
        result = run_cli(["--manifest", str(manifest_path), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dataset_id"] == "test_ds"
        assert data["shard_count"] >= 1

    def test_missing_manifest(self, tmp_path):
        result = run_cli(["--manifest", str(tmp_path / "nonexistent.json")])
        assert result.returncode != 0

    def test_custom_split_name(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        m = create_manifest(
            dataset_id="custom_ds",
            source_id="src",
            source_version="1.0",
            license="mit",
            language="hi",
            split="validation",
            records=50,
            bytes_utf8=25000,
            sha256=_SAMPLE_SHA256,
            processing_config_digest=_SAMPLE_DIGEST,
            registry_digest=_SAMPLE_DIGEST,
            policy_digest=_SAMPLE_DIGEST,
        )
        manifest_path.write_text(json.dumps(m.to_dict(), indent=2))
        result = run_cli(["--manifest", str(manifest_path), "--json"])
        data = json.loads(result.stdout)
        assert data["split"] == "validation"
