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
        check=False,
    )


def _base_args(checkpoint: Path) -> list[str]:
    return [
        "--checkpoint-path",
        str(checkpoint),
        "--output-path",
        "exports/bharat.gguf",
        "--format",
        "gguf",
        "--model-name",
        "bharat-local",
    ]


def test_cli_includes_deterministic_inventory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "z.bin").write_bytes(b"z")
    (checkpoint / "a.json").write_text("{}", encoding="utf-8")

    result = run_cli([*_base_args(checkpoint), "--include-inventory"])

    assert result.returncode == 0
    output = json.loads(result.stdout)
    inventory = output["checkpoint_inventory"]
    assert inventory["checkpoint_path"] == str(checkpoint)
    assert inventory["total_bytes"] == 3
    assert [item["relative_path"] for item in inventory["files"]] == ["a.json", "z.bin"]
    assert all(len(item["sha256"]) == 64 for item in inventory["files"])


def test_cli_omits_inventory_without_flag(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    result = run_cli(_base_args(checkpoint))

    assert result.returncode == 0
    assert "checkpoint_inventory" not in json.loads(result.stdout)


def test_cli_rejects_empty_inventory_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    result = run_cli([*_base_args(checkpoint), "--include-inventory"])

    assert result.returncode != 0
    assert "contains no files" in result.stderr
