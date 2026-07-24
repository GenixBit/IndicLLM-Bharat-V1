from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_serving_control_smoke", *args],
        capture_output=True,
        text=True,
    )


class TestRunServingControlSmokeCLI:
    def test_default_output(self) -> None:
        result = run_cli([])
        assert result.returncode == 0
        assert "Events (" in result.stdout
        assert "Metrics:" in result.stdout

    def test_json_output(self) -> None:
        result = run_cli(["--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "success"
        assert data["event_count"] >= 2

    def test_json_output_with_event_count(self) -> None:
        result = run_cli(["--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["event_count"] >= 2
        assert "metrics" in data

    def test_local_output_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "events.jsonl"
        result = run_cli(["--output", str(output_path)])
        assert result.returncode == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "text_delta" in content or "done" in content

    def test_remote_output_rejected(self) -> None:
        result = run_cli(["--output", "http://example.com/out.jsonl"])
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_https_output_rejected(self) -> None:
        result = run_cli(["--output", "https://example.com/out.jsonl"])
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_ftp_output_rejected(self) -> None:
        result = run_cli(["--output", "ftp://server/out.jsonl"])
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_s3_output_rejected(self) -> None:
        result = run_cli(["--output", "s3://bucket/out.jsonl"])
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_gs_output_rejected(self) -> None:
        result = run_cli(["--output", "gs://bucket/out.jsonl"])
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_metrics_included_in_json_output(self) -> None:
        result = run_cli(["--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        m = data["metrics"]
        assert m["requests_started"] >= 1
        assert m["requests_completed"] >= 1
