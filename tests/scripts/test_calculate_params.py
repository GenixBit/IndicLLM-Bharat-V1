from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "calculate_params.py"
CONFIGS = ROOT / "configs" / "models"
PRODUCTION = ["bharat-350m.yaml", "bharat-1b.yaml", "bharat-3b.yaml", "bharat-7b.yaml"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


class TestCLI:
    def test_human_output(self):
        result = _run(str(CONFIGS / "bharat-350m.yaml"))
        assert result.returncode == 0
        assert "Bharat-350M" in result.stdout
        assert "Parameter breakdown" in result.stdout
        assert "total" in result.stdout

    def test_json_output(self):
        result = _run("--json", str(CONFIGS / "bharat-350m.yaml"))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["model_name"] == "Bharat-350M"
        assert data["actual_parameter_count"] == 347393024
        assert "architecture" in data
        assert "parameter_breakdown" in data

    def test_json_valid(self):
        result = _run("--json", str(CONFIGS / "bharat-1b.yaml"))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["model_name"] == "Bharat-1B"
        assert data["actual_parameter_count"] == 999368704

    def test_weight_memory_included(self):
        result = _run("--json", "--weight-dtype", "bf16", str(CONFIGS / "bharat-350m.yaml"))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "weight_memory" in data
        assert data["weight_memory"]["weight_bytes"] > 0

    def test_kv_cache_included(self):
        result = _run(
            "--json",
            "--weight-dtype",
            "bf16",
            "--batch-size",
            "1",
            "--sequence-length",
            "4096",
            str(CONFIGS / "bharat-7b.yaml"),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "kv_cache" in data
        assert data["kv_cache"]["total_bytes"] > 0

    def test_all_flag(self):
        result = _run("--all", "--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 4

    def test_invalid_file_exit_code(self):
        result = _run("/nonexistent.yaml")
        assert result.returncode != 0

    def test_no_args_shows_help(self):
        result = _run()
        assert result.returncode != 0

    def test_both_config_and_all_rejected(self):
        result = _run("--all", str(CONFIGS / "bharat-350m.yaml"))
        assert result.returncode != 0
