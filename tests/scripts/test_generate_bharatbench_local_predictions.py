from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.generate_bharatbench_local_predictions", *args],
        capture_output=True,
        text=True,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = "\n".join(json.dumps(record) for record in records)
    path.write_text(lines, encoding="utf-8")


def test_remote_checkpoint_rejected(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples_path,
        [{"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"}],
    )

    result = run_cli(
        [
            "--examples",
            str(examples_path),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--checkpoint",
            "https://example.com/checkpoint",
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
        ]
    )

    assert result.returncode != 0
    assert "Remote checkpoint path rejected" in result.stderr


def test_remote_tokenizer_rejected(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples_path,
        [{"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"}],
    )

    result = run_cli(
        [
            "--examples",
            str(examples_path),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--tokenizer",
            "https://example.com/tokenizer.json",
        ]
    )

    assert result.returncode != 0
    assert "Remote tokenizer path rejected" in result.stderr


def test_missing_examples_rejected(tmp_path: Path) -> None:
    result = run_cli(
        [
            "--examples",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
        ]
    )

    assert result.returncode != 0
    assert "Examples file not found" in result.stderr


def test_duplicate_examples_rejected_before_model_loading(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples_path,
        [
            {"example_id": "qa_001", "task_type": "qa", "prompt": "Q?", "expected": "A"},
            {"example_id": "qa_001", "task_type": "qa", "prompt": "Q2?", "expected": "A2"},
        ],
    )

    result = run_cli(
        [
            "--examples",
            str(examples_path),
            "--output",
            str(tmp_path / "predictions.jsonl"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--tokenizer",
            str(tmp_path / "missing-tokenizer.json"),
        ]
    )

    assert result.returncode != 0
    assert "Duplicate example_id" in result.stderr
