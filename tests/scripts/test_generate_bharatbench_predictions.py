from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.generate_bharatbench_predictions", *args],
        capture_output=True,
        text=True,
    )


def run_bharatbench(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_bharatbench", *args],
        capture_output=True,
        text=True,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = "\n".join(json.dumps(r) for r in records)
    path.write_text(lines, encoding="utf-8")


class TestGenerateBharatBenchPredictionsCLI:
    def test_expected_adapter_json_output(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q?",
                    "expected": "A",
                }
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(output_path),
                "--adapter",
                "expected",
                "--json",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["adapter"] == "expected"
        assert data["examples"] == 1
        assert data["predictions"] == 1
        assert (
            output_path.read_text(encoding="utf-8").strip()
            == '{"example_id":"qa_001","prediction":"A"}'
        )

    def test_echo_adapter(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q?",
                    "expected": "A",
                }
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(output_path),
                "--adapter",
                "echo",
            ]
        )
        assert result.returncode == 0
        assert (
            output_path.read_text(encoding="utf-8").strip()
            == '{"example_id":"qa_001","prediction":"Q?"}'
        )

    def test_choice_baseline_adapter(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "cls_001",
                    "task_type": "classification",
                    "prompt": "Language?",
                    "expected": "Hindi",
                    "choices": ["Hindi", "Marathi"],
                }
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(output_path),
                "--adapter",
                "choice-baseline",
            ]
        )
        assert result.returncode == 0
        assert (
            output_path.read_text(encoding="utf-8").strip()
            == '{"example_id":"cls_001","prediction":"Hindi"}'
        )

    def test_invalid_adapter_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q?",
                    "expected": "A",
                }
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(output_path),
                "--adapter",
                "remote-model",
            ]
        )
        assert result.returncode != 0

    def test_missing_examples_exits_nonzero(self, tmp_path: Path) -> None:
        output_path = tmp_path / "predictions.jsonl"
        result = run_cli(
            [
                "--examples",
                str(tmp_path / "missing.jsonl"),
                "--output",
                str(output_path),
                "--adapter",
                "expected",
            ]
        )
        assert result.returncode != 0

    def test_duplicate_examples_exits_nonzero(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        output_path = tmp_path / "predictions.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q?",
                    "expected": "A",
                },
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q2?",
                    "expected": "A2",
                },
            ],
        )

        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(output_path),
                "--adapter",
                "expected",
            ]
        )
        assert result.returncode != 0

    def test_remote_paths_rejected(self) -> None:
        result = run_cli(
            [
                "--examples",
                "https://example.com/examples.jsonl",
                "--output",
                "predictions.jsonl",
                "--adapter",
                "expected",
            ]
        )
        assert result.returncode != 0

    def test_end_to_end_generate_then_run_bharatbench(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        predictions_path = tmp_path / "predictions.jsonl"
        output_dir = tmp_path / "out"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "qa_001",
                    "task_type": "qa",
                    "prompt": "Q?",
                    "expected": "A",
                }
            ],
        )

        generated = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(predictions_path),
                "--adapter",
                "expected",
                "--json",
            ]
        )
        assert generated.returncode == 0

        evaluated = run_bharatbench(
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
        assert evaluated.returncode == 0
        data = json.loads(evaluated.stdout)
        assert data["status"] == "success"
