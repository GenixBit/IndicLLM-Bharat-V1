from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_bharatbench", *args],
        capture_output=True,
        text=True,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = "\n".join(json.dumps(r) for r in records)
    path.write_text(lines, encoding="utf-8")


class TestRunBharatBenchCLI:
    def test_success(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"

        _write_jsonl(
            examples_path,
            [
                {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
            ],
        )
        _write_jsonl(
            predictions_path,
            [
                {"example_id": "qa_001", "prediction": "A"},
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
                "--created-at",
                "2026-07-20T00:00:00Z",
            ]
        )
        assert result.returncode == 0
        assert (output_dir / "bharatbench_report.json").exists()

    def test_json_output(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"

        _write_jsonl(
            examples_path,
            [
                {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
            ],
        )
        _write_jsonl(
            predictions_path,
            [
                {"example_id": "qa_001", "prediction": "A"},
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
                "--created-at",
                "2026-07-20T00:00:00Z",
                "--json",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "success"

    def test_failure_exits_nonzero(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "nonexistent.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"
        _write_jsonl(predictions_path, [{"example_id": "x", "prediction": "y"}])

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0

    def test_missing_prediction_fails(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"
        _write_jsonl(
            examples_path,
            [
                {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
            ],
        )
        _write_jsonl(predictions_path, [])

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0

    def test_duplicate_examples_fails(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"
        _write_jsonl(
            examples_path,
            [
                {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
                {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
            ],
        )
        _write_jsonl(
            predictions_path,
            [
                {"example_id": "qa_001", "prediction": "A"},
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert result.returncode != 0

    def test_tiny_fixtures_load_successfully(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        fixtures_dir = repo_root / "eval_fixtures" / "bharatbench_tiny"
        predictions_path = tmp_path / "predictions.jsonl"
        _write_jsonl(predictions_path, [
            {"example_id": "qa_001", "prediction": "New Delhi"},
            {"example_id": "qa_002", "prediction": "Ganges"},
            {"example_id": "qa_003", "prediction": "Hindi"},
        ])
        output_dir = tmp_path / "out"

        result = run_cli(
            [
                "--examples",
                str(fixtures_dir / "qa.jsonl"),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
                "--created-at",
                "2026-07-20T00:00:00Z",
            ]
        )
        assert result.returncode == 0
