from __future__ import annotations

import json
import subprocess
import sys


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_export_plan", *args],
        capture_output=True,
        text=True,
    )


class TestRunExportPlanCLI:
    def test_safetensors_dry_run(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "exports/bharat.safetensors",
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["export_format"] == "safetensors"
        assert data["dry_run"] is True
        assert data["writer_name"] == "safetensors-dry-run"
        assert data["bytes_written"] == 0

    def test_gguf_dry_run(self) -> None:
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
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["export_format"] == "gguf"
        assert data["writer_name"] == "gguf-dry-run"

    def test_json_output_fields(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "exports/bharat.safetensors",
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "checkpoint_path" in data
        assert "output_path" in data
        assert "export_format" in data
        assert "model_name" in data
        assert "dry_run" in data
        assert "writer_name" in data
        assert "bytes_written" in data

    def test_wrong_suffix_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "exports/bharat.bin",
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode != 0
        assert "error:" in result.stderr

    def test_remote_checkpoint_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "https://example.com/ckpt",
                "--output-path",
                "exports/bharat.gguf",
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode != 0
        assert "Remote checkpoint path rejected" in result.stderr

    def test_remote_output_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "https://example.com/model.gguf",
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_s3_output_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "s3://bucket/model.gguf",
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_normalized_remote_output_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "https:/example.com/model.gguf",
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_empty_model_name_rejected(self) -> None:
        result = run_cli(
            [
                "--checkpoint-path",
                "checkpoints/bharat",
                "--output-path",
                "exports/bharat.gguf",
                "--format",
                "gguf",
                "--model-name",
                "   ",
            ]
        )
        assert result.returncode != 0
        assert "error:" in result.stderr

    def test_missing_required_args(self) -> None:
        result = run_cli([])
        assert result.returncode != 0
        assert "usage:" in result.stderr or "required" in result.stderr
