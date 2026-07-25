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


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    gguf_file = checkpoint / "bharat.gguf"
    gguf_file.write_bytes(b"local-gguf-fixture")
    metadata = tmp_path / "gguf-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "bharat",
                "alignment": 32,
                "tensor_count": 0,
                "output_file": gguf_file.name,
                "metadata": [
                    {"key": "general.name", "value_type": "string"},
                    {"key": "general.file_type", "value_type": "int"},
                ],
            },
        ),
        encoding="utf-8",
    )
    return checkpoint, metadata


def test_gguf_preflight_is_included(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "export.gguf"),
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--gguf-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["gguf_preflight"]["schema_version"] == 1
    assert data["gguf_preflight"]["architecture"] == "bharat"
    assert data["gguf_preflight"]["alignment"] == 32
    assert data["gguf_preflight"]["tensor_count"] == 0
    assert "checkpoint_inventory" not in data


def test_gguf_preflight_rejects_safetensors(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "export.safetensors"),
            "--format",
            "safetensors",
            "--model-name",
            "bharat-local",
            "--gguf-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode != 0
    assert "requires --format gguf" in result.stderr


def test_remote_gguf_metadata_path_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "bharat.gguf").write_bytes(b"fixture")
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "export.gguf"),
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--gguf-metadata-path",
            "https://example.com/gguf-metadata.json",
        ],
    )

    assert result.returncode != 0
    assert "Remote GGUF metadata path rejected" in result.stderr


def test_invalid_gguf_preflight_metadata_returns_nonzero(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["output_file"] = "missing.gguf"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "export.gguf"),
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--gguf-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode != 0
    assert "references missing output file" in result.stderr
