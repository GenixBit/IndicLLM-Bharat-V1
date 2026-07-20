from __future__ import annotations

import json
import sys

import pytest

from scripts.prepare_local_data import main

REAL_TEXT = (
    "The Indian education system has undergone significant changes in recent decades.\n"
    "With the introduction of the National Education Policy 2020, there is a renewed focus on holistic learning.\n"
    "This policy emphasizes critical thinking, experiential learning, and multidisciplinary approaches.\n"
    "It aims to transform India into a vibrant knowledge society and global knowledge superpower.\n"
    "The policy also focuses on early childhood care and education, foundational literacy, and numeracy."
)


def _run(args: list[str]) -> tuple[int, str]:
    from io import StringIO

    old_argv = sys.argv
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.argv = args
    sys.stdout = out = StringIO()
    sys.stderr = err = StringIO()
    try:
        try:
            main()
        except SystemExit as e:
            return e.code, out.getvalue()
        return 0, out.getvalue()
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class TestPrepareLocalDataCLI:
    def test_dry_run(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        code, out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--split",
                "train",
                "--dry-run",
            ]
        )
        assert code == 0
        assert "Total records:" in out
        assert "Accepted records:" in out

    def test_json_output(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        code, out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--split",
                "train",
                "--dry-run",
                "--json",
            ]
        )
        assert code == 0
        data = json.loads(out)
        assert "total_records" in data
        assert "accepted_records" in data
        assert data["shard_count"] == 0

    def test_real_run_creates_shards(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out_dir = str(tmp_path / "out")
        code, out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--split",
                "train",
                "--output-dir",
                out_dir,
            ]
        )
        assert code == 0
        assert "Shards written:" in out
        shard_dir = tmp_path / "out" / "shards"
        assert shard_dir.exists()
        assert len(list(shard_dir.glob("*.jsonl"))) >= 1

    def test_json_output_real(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out_dir = str(tmp_path / "out")
        code, out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--split",
                "train",
                "--output-dir",
                out_dir,
                "--json",
            ]
        )
        assert code == 0
        data = json.loads(out)
        assert data["shard_count"] >= 1

    def test_missing_input_errors(self, tmp_path):
        missing = tmp_path / "nope.txt"
        code, _ = _run(
            [
                "prepare_local_data",
                "--input",
                str(missing),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
            ]
        )
        assert code != 0

    def test_blocklist_option(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("unrelated content here", encoding="utf-8")
        code, out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--dry-run",
                "--blocklist",
                str(blocklist),
            ]
        )
        assert code == 0
