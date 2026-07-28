from __future__ import annotations

from pathlib import Path

from scripts.generate_tokenizer_evidence import _compare_generations


def _run_payload(report_path: Path, decision_path: Path) -> dict:
    return {
        "report": {"report_sha256": "a" * 64},
        "decision": {"acceptance_sha256": "b" * 64},
        "manifest": {"schema_version": "tokenizer-evidence-manifest-v1"},
        "report_path": report_path,
        "decision_path": decision_path,
    }


def test_compare_generations_rejects_different_output_bytes(tmp_path: Path) -> None:
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run1_dir.mkdir()
    run2_dir.mkdir()

    report1 = run1_dir / "evaluation-report.json"
    report2 = run2_dir / "evaluation-report.json"
    decision1 = run1_dir / "acceptance-decision.json"
    decision2 = run2_dir / "acceptance-decision.json"

    report1.write_bytes(b'{"value":1}\n')
    report2.write_bytes(b'{"value":2}\n')
    decision1.write_bytes(b'{"passed":false}\n')
    decision2.write_bytes(b'{"passed":false}\n')

    errors = _compare_generations(
        _run_payload(report1, decision1),
        _run_payload(report2, decision2),
    )

    assert any("evaluation report: byte mismatch" in error for error in errors)
