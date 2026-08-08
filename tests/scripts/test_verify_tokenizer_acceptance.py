from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_tokenizer_acceptance import main

from tests.tokenizer.test_acceptance import _realistic_report, _thresholds


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    report_path = tmp_path / "report.json"
    thresholds_path = tmp_path / "thresholds.json"
    report_path.write_text(json.dumps(_realistic_report()), encoding="utf-8")
    thresholds_path.write_text(
        json.dumps(
            {
                "schema_version": "tokenizer-acceptance-thresholds-v1",
                "status": "provisional",
                "evidence_scope": "synthetic-local-only",
                "thresholds": _thresholds().to_canonical_dict(),
            }
        ),
        encoding="utf-8",
    )
    return report_path, thresholds_path


def test_cli_accepts_valid_local_inputs(tmp_path: Path, capsys) -> None:
    report_path, thresholds_path = _write_inputs(tmp_path)

    assert main(
        [
            "--report",
            str(report_path),
            "--thresholds",
            str(thresholds_path),
            "--tokenizer",
            "bharat-bpe",
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is True
    assert len(output["acceptance_sha256"]) == 64


def test_cli_returns_failure_for_threshold_violation(tmp_path: Path, capsys) -> None:
    report_path, thresholds_path = _write_inputs(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["aggregate"]["bharat-bpe"]["record_count"] = 1
    report["report_sha256"] = ""  # force integrity validation to fail
    report_path.write_text(json.dumps(report), encoding="utf-8")

    try:
        main(
            [
                "--report",
                str(report_path),
                "--thresholds",
                str(thresholds_path),
                "--tokenizer",
                "bharat-bpe",
            ]
        )
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("tampered local report must fail integrity validation")

    assert capsys.readouterr().out == ""
