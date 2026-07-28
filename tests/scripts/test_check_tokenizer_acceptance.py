from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.check_tokenizer_acceptance import main

# ── Helpers ──────────────────────────────────────────────────────────


def _make_report(tmp_path: Path, overrides: dict | None = None) -> Path:
    path = tmp_path / "report.json"
    report: dict = {
        "schema_version": "eval-v1",
        "evaluator_version": "1.0.3",
        "report_sha256": "",
        "input_dataset_sha256": "a" * 64,
        "tokenizer_names": ["bharat-bpe"],
        "tokenizer_fingerprints": {"bharat-bpe": "fp123"},
        "aggregate": {
            "bharat-bpe": {
                "record_count": 12,
                "token_count": 100,
                "unknown_token_count": 0,
                "unknown_token_rate": 0.0,
                "micro_fertility": 1.25,
                "macro_fertility": 1.25,
            }
        },
        "per_language": {
            "bharat-bpe": {
                "en": {"micro_fertility": 1.0, "record_count": 6},
                "hi": {"micro_fertility": 1.5, "record_count": 6},
            }
        },
        "round_trip": {
            "bharat-bpe": {
                "required_pass_rate": 1.0,
                "required_pass_count": 12,
                "canonical_pass_rate": 1.0,
            }
        },
        "byte_coverage": {
            "bharat-bpe": {
                "status": "complete",
                "complete": True,
                "reachable_count": 256,
                "missing_byte_values": [],
            }
        },
        "fragmentation": {"bharat-bpe": {}},
        "comparison": [],
        "failed_records": [],
    }
    if overrides:
        report.update(overrides)
    report["report_sha256"] = _compute_digest(report)
    path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return path


def _compute_digest(report: dict) -> str:
    excluded = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _thresholds_path(tmp_path: Path, overrides: dict | None = None) -> Path:
    path = tmp_path / "thresholds.json"
    thresh: dict = {
        "schema_version": "tokenizer-acceptance-thresholds-v1",
        "thresholds": {
            "min_record_count": 10,
            "min_required_round_trip_rate": 1.0,
            "max_unknown_token_rate": 0.0,
            "require_complete_byte_coverage": True,
            "max_micro_fertility": 2.0,
            "max_language_micro_fertility": 2.0,
        },
    }
    if overrides:
        thresh.update(overrides)
    path.write_text(json.dumps(thresh, sort_keys=True, indent=2))
    return path


def _run_main(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except (ValueError, FileNotFoundError, RuntimeError, FileExistsError):
        # main() raises these directly for validation errors
        return 3


# ── Tests ────────────────────────────────────────────────────────────


def test_successful_passing_report(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 0


def test_threshold_failure_returns_nonzero(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path, {"thresholds": {"min_record_count": 100}})
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 2


def test_corrupted_report_digest_fails_closed(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["aggregate"]["bharat-bpe"]["record_count"] = 999
    report.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    thresh = _thresholds_path(tmp_path)
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 3


def test_unsupported_report_schema(tmp_path: Path) -> None:
    report = _make_report(tmp_path, {"schema_version": "eval-v0"})
    thresh = _thresholds_path(tmp_path)
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 3


def test_corrupted_threshold_config(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = tmp_path / "bad.json"
    thresh.write_text("not valid json")
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 3


def test_dry_run_creates_no_output(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    output = tmp_path / "out.json"
    code = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--dry-run",
            "--execute",
            "--output",
            str(output),
        ]
    )
    # mutually exclusive
    assert code != 0
    assert not output.exists()


def test_execute_writes_canonical_output(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    output = tmp_path / "out.json"
    code = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--execute",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.exists()
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["passed"] is True
    assert "acceptance_sha256" in parsed


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    output = tmp_path / "out.json"
    output.write_text('{"existing": true}')
    code = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--execute",
            "--output",
            str(output),
        ]
    )
    assert code == 3
    assert json.loads(output.read_text(encoding="utf-8")) == {"existing": True}


def test_deterministic_repeated_output(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    code1 = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--execute",
            "--output",
            str(out1),
        ]
    )
    code2 = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--execute",
            "--output",
            str(out2),
        ]
    )
    assert code1 == 0
    assert code2 == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_output_bytes_and_digest_verified(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = _thresholds_path(tmp_path)
    output = tmp_path / "out.json"
    code = _run_main(
        [
            "--report",
            str(report),
            "--thresholds",
            str(thresh),
            "--execute",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    parsed = json.loads(output.read_bytes())
    dig = parsed["acceptance_sha256"]
    # Recompute and check
    excluded = {k: v for k, v in parsed.items() if k != "acceptance_sha256"}
    canonical = json.dumps(excluded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert dig == expected


def test_unsupported_threshold_schema_version(tmp_path: Path) -> None:
    report = _make_report(tmp_path)
    thresh = tmp_path / "thresh.json"
    thresh.write_text(
        json.dumps(
            {
                "schema_version": "tokenizer-acceptance-thresholds-v0",
                "thresholds": {"min_record_count": 10},
            }
        )
    )
    code = _run_main(["--report", str(report), "--thresholds", str(thresh)])
    assert code == 3
