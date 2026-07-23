from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_bharatbench_local_predictions",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = "\n".join(json.dumps(r) for r in records)
    path.write_text(lines, encoding="utf-8")


def _make_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestGenerateBharatBenchLocalPredictionsCLI:
    def test_remote_examples_rejected(self, tmp_path: Path) -> None:
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                "https://example.com/examples.jsonl",
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
            ]
        )
        assert result.returncode != 0
        assert "Remote examples path rejected" in result.stderr

    def test_remote_output_rejected(self, tmp_path: Path) -> None:
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(tmp_path / "examples.jsonl"),
                "--output",
                "s3://bucket/out.jsonl",
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
            ]
        )
        assert result.returncode != 0
        assert "Remote output path rejected" in result.stderr

    def test_remote_checkpoint_rejected(self, tmp_path: Path) -> None:
        result = run_cli(
            [
                "--examples",
                str(tmp_path / "examples.jsonl"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                "https://example.com/model",
                "--tokenizer",
                str(tmp_path / "tokenizer"),
            ]
        )
        assert result.returncode != 0
        assert "Remote checkpoint path rejected" in result.stderr

    def test_remote_tokenizer_rejected(self, tmp_path: Path) -> None:
        result = run_cli(
            [
                "--examples",
                str(tmp_path / "examples.jsonl"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--tokenizer",
                "gs://bucket/tokenizer",
            ]
        )
        assert result.returncode != 0
        assert "Remote tokenizer path rejected" in result.stderr

    def test_missing_examples_file(self, tmp_path: Path) -> None:
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(tmp_path / "nonexistent.jsonl"),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
            ]
        )
        assert result.returncode != 0
        assert "Examples file not found" in result.stderr

    def test_duplicate_example_ids_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        _write_jsonl(
            examples_path,
            [
                {
                    "example_id": "dup_001",
                    "task_type": "qa",
                    "prompt": "Q1?",
                    "expected": "A1",
                },
                {
                    "example_id": "dup_001",
                    "task_type": "qa",
                    "prompt": "Q2?",
                    "expected": "A2",
                },
            ],
        )
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
            ]
        )
        assert result.returncode != 0
        assert "Duplicate example_id" in result.stderr

    def test_missing_checkpoint_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
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
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(tmp_path / "nonexistent_ckpt"),
                "--tokenizer",
                str(tok),
            ]
        )
        assert result.returncode != 0
        assert "Checkpoint not found" in result.stderr

    def test_missing_tokenizer_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
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
        ckpt = _make_dir(tmp_path / "checkpoint")
        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tmp_path / "nonexistent_tok"),
            ]
        )
        assert result.returncode != 0
        assert "Tokenizer not found" in result.stderr

    def test_invalid_max_new_tokens_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
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
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
                "--max-new-tokens",
                "0",
            ]
        )
        assert result.returncode != 0
        assert "max_new_tokens must be >= 1" in result.stderr

    def test_negative_max_new_tokens_rejected(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
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
        ckpt = _make_dir(tmp_path / "checkpoint")
        tok = _make_dir(tmp_path / "tokenizer")
        result = run_cli(
            [
                "--examples",
                str(examples_path),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--checkpoint",
                str(ckpt),
                "--tokenizer",
                str(tok),
                "--max-new-tokens",
                "-1",
            ]
        )
        assert result.returncode != 0
        assert "max_new_tokens must be >= 1" in result.stderr
