from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.compute_data_stats", *args],
        capture_output=True,
        text=True,
    )


class TestComputeStatsCLI:
    def test_single_text_file(self, tmp_path):
        text_file = tmp_path / "input.txt"
        text_file.write_text(
            "This is a test document with enough content for quality checks.\n"
            "It has multiple lines so it passes quality thresholds.\n"
        )
        result = run_cli(["--input", str(text_file)])
        assert result.returncode == 0
        assert "Records:" in result.stdout
        assert "Accepted:" in result.stdout

    def test_json_output(self, tmp_path):
        text_file = tmp_path / "input.txt"
        text_file.write_text("Sample document for testing purposes.\nMultiple lines here.\n")
        result = run_cli(["--input", str(text_file), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["record_count"] >= 1
        assert "accepted_count" in data
        assert "rejected_count" in data

    def test_missing_input(self, tmp_path):
        result = run_cli(["--input", str(tmp_path / "nonexistent.txt")])
        assert result.returncode != 0

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = run_cli(["--input", str(empty_dir)])
        assert result.returncode != 0
        assert "no text records" in result.stderr

    def test_jsonl_input(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            "First document with enough text for quality scoring.\n"
        )
        result = run_cli(["--input", str(jsonl_file), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["record_count"] == 1

    def test_multiple_files_in_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text(
            "Document A with proper content.\nSecond line for quality.\n"
        )
        (tmp_path / "b.txt").write_text(
            "Document B with proper content.\nSecond line for quality.\n"
        )
        result = run_cli(["--input", str(tmp_path), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["record_count"] == 2
