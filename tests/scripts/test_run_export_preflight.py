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
    shard = checkpoint / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"local-shard-fixture")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_tensor_bytes": 8,
                "tensors": [
                    {
                        "name": "transformer.weight",
                        "shape": [4],
                        "dtype": "BF16",
                        "shard": shard.name,
                        "size_bytes": 8,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    return checkpoint, metadata


def test_safetensors_preflight_is_included(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "bharat.safetensors"),
            "--format",
            "safetensors",
            "--model-name",
            "bharat-local",
            "--safetensors-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["safetensors_preflight"]["schema_version"] == 1
    assert data["safetensors_preflight"]["tensor_count"] == 1
    assert data["safetensors_preflight"]["total_tensor_bytes"] == 8
    assert "checkpoint_inventory" not in data


def test_safetensors_preflight_rejects_gguf(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "bharat.gguf"),
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--safetensors-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode != 0
    assert "requires --format safetensors" in result.stderr


def test_remote_safetensors_metadata_path_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "bharat.safetensors"),
            "--format",
            "safetensors",
            "--model-name",
            "bharat-local",
            "--safetensors-metadata-path",
            "https://example.com/metadata.json",
        ],
    )

    assert result.returncode != 0
    assert "Remote safetensors metadata path rejected" in result.stderr


def test_invalid_preflight_metadata_returns_nonzero(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["total_tensor_bytes"] = 99
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    result = run_cli(
        [
            "--checkpoint-path",
            str(checkpoint),
            "--output-path",
            str(tmp_path / "bharat.safetensors"),
            "--format",
            "safetensors",
            "--model-name",
            "bharat-local",
            "--safetensors-metadata-path",
            str(metadata),
        ],
    )

    assert result.returncode != 0
    assert "does not match tensor sum" in result.stderr
