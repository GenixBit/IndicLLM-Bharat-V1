from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_production_tokenizer_evidence import main
from tests.tokenizer.evidence_fixtures import (
    build_acceptance_decision,
    build_bpe_tokenizer,
    build_input_jsonl,
    build_production_thresholds,
    canonical_bytes,
    compute_real_report,
)


def _canonical(value: object) -> bytes:
    return canonical_bytes(value)


def _build_bpe_tokenizer(tmp_path: Path) -> Path:
    return build_bpe_tokenizer(tmp_path, name="tokenizer.json")


def _build_input_jsonl(tmp_path: Path) -> Path:
    return build_input_jsonl(tmp_path, name="input.jsonl")


def _build_thresholds(tmp_path: Path) -> Path:
    return build_production_thresholds(tmp_path, name="thresholds.json")


def _build_report(tmp_path: Path, tokenizer_name: str, input_path: Path) -> Path:
    return compute_real_report(tmp_path, tmp_path / "tokenizer.json", input_path, tokenizer_name)


def _build_decision(
    tmp_path: Path, report_path: Path, thresholds_path: Path, tokenizer_name: str
) -> Path:
    return build_acceptance_decision(tmp_path, report_path, thresholds_path, tokenizer_name)


@pytest.fixture
def cli_fixtures(tmp_path: Path) -> dict[str, Path]:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_thresholds(tmp_path)
    report_path = _build_report(tmp_path, "test-bpe", input_path)
    decision_path = _build_decision(tmp_path, report_path, thresholds_path, "test-bpe")
    return {
        "tokenizer_path": tokenizer_path,
        "input_path": input_path,
        "thresholds_path": thresholds_path,
        "report_path": report_path,
        "decision_path": decision_path,
    }


def _build_args(
    tmp_path: Path,
    fixtures: dict[str, Path],
    output_name: str = "manifest.json",
) -> list[str]:
    return [
        "--evidence-root",
        str(tmp_path),
        "--repository-commit-sha",
        "a" * 40,
        "--tokenizer",
        str(fixtures["tokenizer_path"]),
        "--evaluation-input",
        str(fixtures["input_path"]),
        "--evaluation-report",
        str(fixtures["report_path"]),
        "--acceptance-decision",
        str(fixtures["decision_path"]),
        "--threshold-configuration",
        str(fixtures["thresholds_path"]),
        "--generating-command",
        "test-command",
        "--output",
        str(tmp_path / output_name),
    ]


# -- CLI success --


def test_cli_success(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    rc = main(_build_args(tmp_path, cli_fixtures))
    assert rc == 0
    out = tmp_path / "manifest.json"
    assert out.exists()


# -- CLI validation failure --


def test_cli_validation_failure(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    args = _build_args(tmp_path, cli_fixtures)
    bad_report = tmp_path / "bad_report.json"
    bad_report.write_text("not json", encoding="utf-8")
    idx = args.index(str(cli_fixtures["report_path"]))
    args[idx] = str(bad_report)
    rc = main(args)
    assert rc == 2


# -- CLI existing-output failure --


def test_cli_existing_output_failure(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    out_file = tmp_path / "manifest.json"
    out_file.write_text("existing", encoding="utf-8")
    args = _build_args(tmp_path, cli_fixtures)
    rc = main(args)
    assert rc == 3
    assert out_file.read_text(encoding="utf-8") == "existing"


# -- CLI outputs manifest_sha256 --


def test_cli_outputs_digest(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    cap = _capture_main(_build_args(tmp_path, cli_fixtures))
    assert "manifest_sha256" in cap


def _capture_main(argv: list[str]) -> str:
    import io
    import sys as _sys

    old_out = _sys.stdout
    try:
        _sys.stdout = io.StringIO()
        rc = main(argv)
        assert rc == 0
        return _sys.stdout.getvalue()
    finally:
        _sys.stdout = old_out


# -- Offline execution --


def test_cli_offline(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    rc = main(_build_args(tmp_path, cli_fixtures))
    assert rc == 0


# -- Determinism across CLI invocations --


def test_cli_deterministic(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    args1 = _build_args(tmp_path, cli_fixtures, output_name="out1.json")
    args2 = _build_args(tmp_path, cli_fixtures, output_name="out2.json")
    assert main(args1) == 0
    assert main(args2) == 0
    assert out1.read_bytes() == out2.read_bytes()
