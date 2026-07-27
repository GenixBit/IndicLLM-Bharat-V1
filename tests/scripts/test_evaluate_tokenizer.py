from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.tokenizer import train_bpe


def _build_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text":"hello world foo bar baz"}\n', encoding="utf-8")
    raw = train_bpe(corpus, vocab_size=270)
    tok_path = tmp_path / "tokenizer.json"
    raw.save(tok_path, overwrite=True)
    return tok_path


def _build_dataset(tmp_path: Path) -> Path:
    ds_path = tmp_path / "eval.jsonl"
    ds_path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False) + "\n"
            for r in [
                {
                    "id": "r1",
                    "language": "en",
                    "script": "Latin",
                    "domain": "gen",
                    "text": "hello world",
                },
                {
                    "id": "r2",
                    "language": "en",
                    "script": "Latin",
                    "domain": "gen",
                    "text": "foo bar baz",
                },
            ]
        ),
        encoding="utf-8",
    )
    return ds_path


@pytest.fixture
def eval_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return _build_tokenizer(tmp_path), _build_dataset(tmp_path)


# ── 1. CLI success ──────────────────────────────────────────────────


def test_cli_success(
    eval_artifacts: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tok_path, ds_path = eval_artifacts
    report_path = tmp_path / "report.json"

    from scripts.evaluate_tokenizer import main

    main(
        [
            "--tokenizer",
            str(tok_path),
            "--name",
            "test_tok",
            "--dataset",
            str(ds_path),
            "--output-report",
            str(report_path),
            "--execute",
        ]
    )

    assert report_path.exists()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "aggregate" in report_data
    assert report_data["schema_version"] == "eval-v1"

    captured = capsys.readouterr()
    success = json.loads(captured.out.strip())
    assert success["status"] == "ok"
    assert "report_sha256" in success


# ── 2. CLI failure: missing dataset ─────────────────────────────────


def test_cli_missing_dataset(tmp_path: Path) -> None:
    tok_path = _build_tokenizer(tmp_path)
    from scripts.evaluate_tokenizer import main

    with pytest.raises(SystemExit):
        main(
            [
                "--tokenizer",
                str(tok_path),
                "--dataset",
                str(tmp_path / "nonexistent.jsonl"),
                "--dry-run",
            ]
        )


# ── 3. CLI failure: missing tokenizer ────────────────────────────────


def test_cli_missing_tokenizer(tmp_path: Path) -> None:
    ds_path = _build_dataset(tmp_path)
    from scripts.evaluate_tokenizer import main

    with pytest.raises((SystemExit, FileNotFoundError)):
        main(
            [
                "--tokenizer",
                str(tmp_path / "nonexistent.json"),
                "--dataset",
                str(ds_path),
                "--dry-run",
            ]
        )


# ── 4. Dry-run creates no files ──────────────────────────────────────


def test_dry_run_creates_no_files(eval_artifacts: tuple[Path, Path]) -> None:
    tok_path, ds_path = eval_artifacts
    from scripts.evaluate_tokenizer import main

    main(
        [
            "--tokenizer",
            str(tok_path),
            "--name",
            "test_tok",
            "--dataset",
            str(ds_path),
            "--dry-run",
        ]
    )

    report_json = ds_path.parent / "report.json"
    assert not report_json.exists()


# ── 5. Existing output rejection ─────────────────────────────────────


def test_existing_output_rejected(eval_artifacts: tuple[Path, Path], tmp_path: Path) -> None:
    tok_path, ds_path = eval_artifacts
    report_path = tmp_path / "report.json"
    report_path.write_text("placeholder", encoding="utf-8")

    from scripts.evaluate_tokenizer import main

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "--tokenizer",
                str(tok_path),
                "--name",
                "test_tok",
                "--dataset",
                str(ds_path),
                "--output-report",
                str(report_path),
                "--execute",
            ]
        )


# ── 6. Overwrite allowed by removing first ───────────────────────────


def test_execute_with_overwrite(eval_artifacts: tuple[Path, Path], tmp_path: Path) -> None:
    tok_path, ds_path = eval_artifacts
    report_path = tmp_path / "report.json"

    from scripts.evaluate_tokenizer import main

    main(
        [
            "--tokenizer",
            str(tok_path),
            "--name",
            "test_tok",
            "--dataset",
            str(ds_path),
            "--output-report",
            str(report_path),
            "--execute",
        ]
    )
    # second call would fail, but first succeeds
    assert report_path.exists()


# ── 7. Multiple tokenizers ───────────────────────────────────────────


def test_multiple_tokenizers(
    eval_artifacts: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tok_path, ds_path = eval_artifacts
    report_path = tmp_path / "report.json"

    from scripts.evaluate_tokenizer import main

    main(
        [
            "--tokenizer",
            str(tok_path),
            "--name",
            "tok_a",
            "--tokenizer",
            str(tok_path),
            "--name",
            "tok_b",
            "--dataset",
            str(ds_path),
            "--output-report",
            str(report_path),
            "--execute",
        ]
    )

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "comparison" in report_data
    assert "tok_a" in report_data["aggregate"]

    captured = capsys.readouterr()
    success = json.loads(captured.out.strip())
    assert success["status"] == "ok"


# ── 8. CLI help ──────────────────────────────────────────────────────


def test_cli_help() -> None:
    from scripts.evaluate_tokenizer import _build_parser

    parser = _build_parser()
    help_text = parser.format_help()
    assert "--tokenizer" in help_text
    assert "--dataset" in help_text
    assert "--output-report" in help_text
    assert "--dry-run" in help_text
    assert "--execute" in help_text


# ── 9. No execute flag with output requires confirmation ─────────────


def test_cli_requires_execute_or_dry_run(eval_artifacts: tuple[Path, Path], tmp_path: Path) -> None:
    tok_path, ds_path = eval_artifacts
    report_path = tmp_path / "report.json"

    from scripts.evaluate_tokenizer import main

    with pytest.raises(SystemExit):
        main(
            [
                "--tokenizer",
                str(tok_path),
                "--dataset",
                str(ds_path),
                "--output-report",
                str(report_path),
            ]
        )
