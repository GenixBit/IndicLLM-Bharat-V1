from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "train_tokenizer.py"), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"text": "hello world"}) + "\n",
        json.dumps({"text": "bpe tokenizer"}) + "\n",
        json.dumps({"text": "deterministic"}) + "\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_help_succeeds() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "train" in result.stdout.lower()


def test_missing_corpus_fails(tmp_path: Path) -> None:
    result = _run(
        "--corpus",
        str(tmp_path / "nope.jsonl"),
        "--vocab-size",
        "270",
        "-o",
        str(tmp_path / "out.json"),
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


def test_train_basic(corpus_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "tokenizer.json"
    result = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output))
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "vocab" in data
    assert "merges" in data
    assert "special_tokens" in data


def test_train_special_tokens(corpus_file: Path, tmp_path: Path) -> None:
    special = '{"<pad>":0,"<unk>":1}'
    output = tmp_path / "tokenizer.json"
    result = _run(
        "--corpus",
        str(corpus_file),
        "--vocab-size",
        "270",
        "-o",
        str(output),
        "--special-tokens",
        special,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["special_tokens"]["<pad>"] == 0
    assert data["special_tokens"]["<unk>"] == 1


def test_train_invalid_special_tokens_json(corpus_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "tokenizer.json"
    result = _run(
        "--corpus",
        str(corpus_file),
        "--vocab-size",
        "270",
        "-o",
        str(output),
        "--special-tokens",
        "not-json",
    )
    assert result.returncode != 0


def test_train_deterministic(corpus_file: Path, tmp_path: Path) -> None:
    output1 = tmp_path / "t1.json"
    output2 = tmp_path / "t2.json"
    r1 = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output1))
    assert r1.returncode == 0
    r2 = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output2))
    assert r2.returncode == 0
    d1 = json.loads(output1.read_text(encoding="utf-8"))
    d2 = json.loads(output2.read_text(encoding="utf-8"))
    assert d1 == d2


def test_train_deterministic_with_delay(corpus_file: Path, tmp_path: Path) -> None:
    output1 = tmp_path / "t1.json"
    output2 = tmp_path / "t2.json"
    r1 = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output1))
    assert r1.returncode == 0
    time.sleep(3)
    r2 = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output2))
    assert r2.returncode == 0
    d1 = json.loads(output1.read_text(encoding="utf-8"))
    d2 = json.loads(output2.read_text(encoding="utf-8"))
    assert d1 == d2


def test_train_output_contains_hash(corpus_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "tokenizer.json"
    result = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output))
    assert result.returncode == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "tokenizer_hash" in data
    assert len(data["tokenizer_hash"]) == 64


def test_train_output_directory_created(corpus_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "subdir" / "nested" / "tokenizer.json"
    result = _run("--corpus", str(corpus_file), "--vocab-size", "270", "-o", str(output))
    assert result.returncode == 0
    assert output.exists()
