from __future__ import annotations

import json
import sys

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
    sys.stderr = StringIO()
    try:
        ret = main()
        return (ret if ret is not None else 0), out.getvalue()
    except SystemExit as e:
        return (e.code if e.code is not None else 0), out.getvalue()
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

    def test_created_at_reproducible(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")
        ts = "2026-07-20T12:00:00Z"
        code1, out1_out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test_repro",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--output-dir",
                out1,
                "--created-at",
                ts,
            ]
        )
        assert code1 == 0
        report1_path = tmp_path / "out1" / "preparation_report.json"
        d1 = json.loads(report1_path.read_text(encoding="utf-8"))

        code2, out2_out = _run(
            [
                "prepare_local_data",
                "--input",
                str(f),
                "--source-id",
                "test_repro",
                "--source-version",
                "1.0",
                "--license",
                "cc-by-4.0",
                "--language",
                "en",
                "--output-dir",
                out2,
                "--created-at",
                ts,
            ]
        )
        assert code2 == 0
        report2_path = tmp_path / "out2" / "preparation_report.json"
        d2 = json.loads(report2_path.read_text(encoding="utf-8"))

        assert d1["manifest_digest"] == d2["manifest_digest"]

    def test_created_at_invalid_format(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        code, _ = _run(
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
                "--created-at",
                "not-a-valid-timestamp",
            ]
        )
        assert code != 0

    def test_json_output_is_valid_json(self, tmp_path):
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
                "--dry-run",
                "--json",
            ]
        )
        assert code == 0
        json.loads(out)
