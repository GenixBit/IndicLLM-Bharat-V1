from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_EVIDENCE_DIR = Path("evidence/tokenizer/milestone-6-1-synthetic")
_REPO_ROOT = Path(__file__).resolve().parent.parent

_TOKENIZER = "tests/fixtures/tiny_bpe_tokenizer.json"
_DATASET = "tests/fixtures/tokenizer_eval/all.jsonl"
_THRESHOLDS = "configs/tokenizers/bpe-64k-acceptance.json"
_TOKENIZER_NAME = "tiny-bpe"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(tmp: Path) -> dict[str, Any]:
    report_path = tmp / "evaluation-report.json"
    decision_path = tmp / "acceptance-decision.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_tokenizer",
            "--tokenizer",
            str(_REPO_ROOT / _TOKENIZER),
            "--name",
            _TOKENIZER_NAME,
            "--dataset",
            str(_REPO_ROOT / _DATASET),
            "--execute",
            "--output-report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"evaluate_tokenizer failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
        )

    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_tokenizer_acceptance",
            "--report",
            str(report_path),
            "--thresholds",
            str(_REPO_ROOT / _THRESHOLDS),
            "--execute",
            "--output",
            str(decision_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result2.returncode not in (0, 2):
        raise RuntimeError(
            f"check_tokenizer_acceptance failed (code={result2.returncode}):"
            f"\nstdout:{result2.stdout}\nstderr:{result2.stderr}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = _build_manifest(report_path, decision_path)

    return {
        "report": report,
        "decision": json.loads(decision_path.read_text(encoding="utf-8")),
        "manifest": manifest,
        "report_path": report_path,
        "decision_path": decision_path,
    }


def _build_manifest(
    report_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    return {
        "schema_version": "tokenizer-evidence-manifest-v1",
        "evidence_scope": "synthetic-local-only",
        "status": "provisional",
        "tokenizer": {
            "artifact_path": _TOKENIZER,
            "artifact_sha256": _digest(_REPO_ROOT / _TOKENIZER),
            "fingerprint": decision.get("tokenizer_fingerprint", ""),
        },
        "evaluation_fixture": {
            "path": _DATASET,
            "sha256": _digest(_REPO_ROOT / _DATASET),
        },
        "threshold_configuration": {
            "path": _THRESHOLDS,
            "sha256": _digest(_REPO_ROOT / _THRESHOLDS),
            "thresholds_sha256": decision.get("thresholds_sha256", ""),
            "configuration_sha256": decision.get("threshold_configuration_sha256", ""),
        },
        "evaluation_report": {
            "path": "evaluation-report.json",
            "sha256": _digest(report_path),
            "report_sha256": decision.get("input_report_sha256", ""),
            "input_dataset_sha256": decision.get("input_dataset_sha256", ""),
        },
        "acceptance_decision": {
            "path": "acceptance-decision.json",
            "sha256": _digest(decision_path),
            "acceptance_sha256": decision.get("acceptance_sha256", ""),
            "input_report_sha256": decision.get("input_report_sha256", ""),
            "tokenizer_name": _TOKENIZER_NAME,
            "tokenizer_fingerprint": decision.get("tokenizer_fingerprint", ""),
            "passed": decision.get("passed", False),
        },
        "generating_commands": [
            "python -m scripts.evaluate_tokenizer "
            "--tokenizer toys/tiny_bpe_tokenizer.json "
            "--name tiny-bpe "
            "--dataset fixtures/tokenizer_eval/all.jsonl "
            "--execute --output-report <tmp>/evaluation-report.json",
            "python -m scripts.check_tokenizer_acceptance "
            "--report <tmp>/evaluation-report.json "
            "--thresholds configs/tokenizers/bpe-64k-acceptance.json "
            "--execute --output <tmp>/acceptance-decision.json",
        ],
    }


def _compare_byte(label: str, path_a: Path, path_b: Path, errors: list[str]) -> None:
    a = path_a.read_bytes()
    b = path_b.read_bytes()
    if a != b:
        errors.append(f"{label}: byte mismatch ({len(a)} vs {len(b)} bytes)")


def _canonical_digest(obj: dict[str, Any], exclude: str | None = None) -> str:
    payload = {k: v for k, v in obj.items() if k != exclude}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic regeneration of synthetic tokenizer evidence"
    )
    parser.add_argument(
        "--verify-committed",
        action="store_true",
        help="Verify generated evidence matches committed evidence",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        run1 = _generate(tmp)

        run2_dir = tmp / "run2"
        run2_dir.mkdir()
        for f in ["evaluation-report.json", "acceptance-decision.json"]:
            (run1["report_path"].parent / f).rename(run2_dir / f)
        run2 = _generate(tmp)

        errors: list[str] = []

        # Compare byte-for-byte
        _compare_byte("evaluation report", run1["report_path"], run2["report_path"], errors)
        _compare_byte("acceptance decision", run1["decision_path"], run2["decision_path"], errors)

        # Compare manifests
        m1 = json.dumps(run1["manifest"], sort_keys=True, indent=2)
        m2 = json.dumps(run2["manifest"], sort_keys=True, indent=2)
        if m1 != m2:
            errors.append("manifest payload mismatch between generations")

        # Compare internal digests
        r1_digest = _canonical_digest(run1["report"], "report_sha256")
        r2_digest = _canonical_digest(run2["report"], "report_sha256")
        if r1_digest != r2_digest:
            errors.append("report internal digest mismatch between generations")
        if run1["report"].get("report_sha256") != run2["report"].get("report_sha256"):
            errors.append("report_sha256 mismatch between generations")

        d1_digest = _canonical_digest(run1["decision"], "acceptance_sha256")
        d2_digest = _canonical_digest(run2["decision"], "acceptance_sha256")
        if d1_digest != d2_digest:
            errors.append("decision internal digest mismatch between generations")
        if run1["decision"].get("acceptance_sha256") != run2["decision"].get("acceptance_sha256"):
            errors.append("acceptance_sha256 mismatch between generations")

        if errors:
            for err in errors:
                print(f"error: {err}", file=sys.stderr)
            return 1

        print("Generation is deterministic: byte-identical across two runs")

        if args.verify_committed:
            committed_report = _EVIDENCE_DIR / "evaluation-report.json"
            committed_decision = _EVIDENCE_DIR / "acceptance-decision.json"
            committed_manifest = _EVIDENCE_DIR / "manifest.json"

            if not committed_report.is_file():
                print("error: committed evaluation report not found", file=sys.stderr)
                return 1

            ver_errors: list[str] = []
            _compare_byte(
                "committed vs generated report", committed_report, run1["report_path"], ver_errors
            )
            _compare_byte(
                "committed vs generated decision",
                committed_decision,
                run1["decision_path"],
                ver_errors,
            )

            cm = json.loads(committed_manifest.read_text(encoding="utf-8"))

            cmp_manifest_fields = [
                ("schema_version", "schema_version"),
                ("evidence_scope", "evidence_scope"),
                ("status", "status"),
                ("tokenizer.artifact_path", "tokenizer", "artifact_path"),
                ("tokenizer.artifact_sha256", "tokenizer", "artifact_sha256"),
                ("tokenizer.fingerprint", "tokenizer", "fingerprint"),
                ("evaluation_fixture.path", "evaluation_fixture", "path"),
                ("evaluation_fixture.sha256", "evaluation_fixture", "sha256"),
                (
                    "threshold_configuration.path",
                    "threshold_configuration",
                    "path",
                ),
                (
                    "threshold_configuration.sha256",
                    "threshold_configuration",
                    "sha256",
                ),
                (
                    "threshold_configuration.thresholds_sha256",
                    "threshold_configuration",
                    "thresholds_sha256",
                ),
                (
                    "threshold_configuration.configuration_sha256",
                    "threshold_configuration",
                    "configuration_sha256",
                ),
                ("evaluation_report.path", "evaluation_report", "path"),
                ("evaluation_report.sha256", "evaluation_report", "sha256"),
                (
                    "evaluation_report.report_sha256",
                    "evaluation_report",
                    "report_sha256",
                ),
                (
                    "evaluation_report.input_dataset_sha256",
                    "evaluation_report",
                    "input_dataset_sha256",
                ),
                (
                    "acceptance_decision.path",
                    "acceptance_decision",
                    "path",
                ),
                (
                    "acceptance_decision.sha256",
                    "acceptance_decision",
                    "sha256",
                ),
                (
                    "acceptance_decision.acceptance_sha256",
                    "acceptance_decision",
                    "acceptance_sha256",
                ),
                (
                    "acceptance_decision.input_report_sha256",
                    "acceptance_decision",
                    "input_report_sha256",
                ),
                (
                    "acceptance_decision.tokenizer_name",
                    "acceptance_decision",
                    "tokenizer_name",
                ),
                (
                    "acceptance_decision.tokenizer_fingerprint",
                    "acceptance_decision",
                    "tokenizer_fingerprint",
                ),
                ("acceptance_decision.passed", "acceptance_decision", "passed"),
            ]

            for entry in cmp_manifest_fields:
                label = entry[0]
                keys = entry[1:]
                cv = cm
                gv = run1["manifest"]
                for k in keys:
                    cv = cv.get(k, {}) if isinstance(cv, dict) else None
                    gv = gv.get(k, {}) if isinstance(gv, dict) else None
                if cv != gv:
                    ver_errors.append(
                        f"committed vs generated manifest {label}: "
                        f"committed={cv!r}, generated={gv!r}"
                    )

            if ver_errors:
                for err in ver_errors:
                    print(f"error: {err}", file=sys.stderr)
                return 1

            print("Generated evidence matches committed evidence")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
