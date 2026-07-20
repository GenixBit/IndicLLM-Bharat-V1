from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bharat.data.manifest import create_manifest


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.validate_data_manifest", *args],
        capture_output=True,
        text=True,
    )


_SAMPLE_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_SAMPLE_DIGEST = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"


def _write_manifest(path: Path, overrides: dict | None = None) -> dict:
    m = create_manifest(
        dataset_id="test_ds",
        source_id="test_src",
        source_version="1.0.0",
        license="cc-by-4.0",
        language="en",
        split="train",
        records=100,
        bytes_utf8=50000,
        sha256=_SAMPLE_SHA256,
        processing_config_digest=_SAMPLE_DIGEST,
        registry_digest=_SAMPLE_DIGEST,
        policy_digest=_SAMPLE_DIGEST,
    )
    data = m.to_dict()
    if overrides:
        data.update(overrides)
    path.write_text(json.dumps(data, indent=2))
    return data


class TestValidateManifestCLI:
    def test_valid_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(manifest_path)
        result = run_cli(["--manifest", str(manifest_path)])
        assert result.returncode == 0
        assert "Manifest valid" in result.stdout

    def test_valid_manifest_json(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(manifest_path)
        result = run_cli(["--manifest", str(manifest_path), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "valid"
        assert data["dataset_id"] == "test_ds"

    def test_missing_file(self, tmp_path):
        result = run_cli(["--manifest", str(tmp_path / "nonexistent.json")])
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_invalid_json(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{invalid}")
        result = run_cli(["--manifest", str(manifest_path)])
        assert result.returncode != 0

    def test_invalid_manifest_schema(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(manifest_path, overrides={"manifest_version": "0.5"})
        result = run_cli(["--manifest", str(manifest_path)])
        assert result.returncode != 0

    def test_digest_mismatch_warning(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(
            manifest_path,
            overrides={
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            },
        )
        result = run_cli(["--manifest", str(manifest_path)])
        assert result.returncode == 0
        assert "digest mismatch" in result.stdout

    def test_json_failure_output(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(manifest_path, overrides={"manifest_version": "0.5"})
        result = run_cli(["--manifest", str(manifest_path), "--json"])
        data = json.loads(result.stdout)
        assert data["status"] == "invalid"
        assert len(data["errors"]) > 0
