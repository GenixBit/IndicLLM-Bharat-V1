from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_export_plan", *args],
        capture_output=True,
        text=True,
    )


def test_cli_writes_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "export-manifest.json"
    result = run_cli(
        [
            "--checkpoint-path",
            "checkpoints/bharat",
            "--output-path",
            "exports/bharat.gguf",
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--manifest-path",
            str(manifest_path),
        ]
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert output["manifest_path"] == str(manifest_path)
    assert output["manifest_schema_version"] == "1.0"
    assert manifest["export_format"] == "gguf"
    assert manifest["dry_run"] is True


def test_cli_rejects_remote_manifest_path() -> None:
    result = run_cli(
        [
            "--checkpoint-path",
            "checkpoints/bharat",
            "--output-path",
            "exports/bharat.gguf",
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--manifest-path",
            "s3://bucket/export-manifest.json",
        ]
    )

    assert result.returncode != 0
    assert "Remote manifest path rejected" in result.stderr
