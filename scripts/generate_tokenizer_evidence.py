from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.validate_tokenizer_evidence import validate_evidence

_EVIDENCE_DIR = Path("evidence/tokenizer/milestone-6-1-synthetic")
_REPO_ROOT = Path(__file__).resolve().parent.parent

_TOKENIZER = "tests/fixtures/tiny_bpe_tokenizer.json"
_DATASET = "tests/fixtures/tokenizer_eval/all.jsonl"
_THRESHOLDS = "configs/tokenizers/bpe-64k-acceptance.json"
_TOKENIZER_NAME = "tiny-bpe"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _canonical_digest(obj: dict[str, Any], exclude: str | None = None) -> str:
    payload = {k: v for k, v in obj.items() if k != exclude}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "evaluation-report.json"
    decision_path = output_dir / "acceptance-decision.json"

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
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    manifest = _build_manifest(decision, report_path, decision_path)

    return {
        "report": report,
        "decision": decision,
        "manifest": manifest,
        "report_path": report_path,
        "decision_path": decision_path,
        "output_dir": output_dir,
    }


def _build_manifest(
    decision: dict[str, Any],
    report_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
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
            {
                "module": "scripts.evaluate_tokenizer",
                "arguments": [
                    "--tokenizer",
                    "tests/fixtures/tiny_bpe_tokenizer.json",
                    "--name",
                    "tiny-bpe",
                    "--dataset",
                    "tests/fixtures/tokenizer_eval/all.jsonl",
                    "--execute",
                    "--output-report",
                    "<run-dir>/evaluation-report.json",
                ],
            },
            {
                "module": "scripts.check_tokenizer_acceptance",
                "arguments": [
                    "--report",
                    "<run-dir>/evaluation-report.json",
                    "--thresholds",
                    "configs/tokenizers/bpe-64k-acceptance.json",
                    "--execute",
                    "--output",
                    "<run-dir>/acceptance-decision.json",
                ],
            },
        ],
    }


def _compare_byte(label: str, path_a: Path, path_b: Path, errors: list[str]) -> None:
    a = path_a.read_bytes()
    b = path_b.read_bytes()
    if a != b:
        errors.append(f"{label}: byte mismatch ({len(a)} vs {len(b)} bytes)")


def _compare_generations(run1: dict[str, Any], run2: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    _compare_byte("evaluation report", run1["report_path"], run2["report_path"], errors)
    _compare_byte("acceptance decision", run1["decision_path"], run2["decision_path"], errors)
    if _canonical_bytes(run1["manifest"]) != _canonical_bytes(run2["manifest"]):
        errors.append("manifest payload mismatch between generations")

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
    if run1["decision"].get("input_dataset_sha256") != run2["decision"].get("input_dataset_sha256"):
        errors.append("input_dataset_sha256 mismatch between generations")
    if run1["decision"].get("threshold_configuration_sha256") != run2["decision"].get(
        "threshold_configuration_sha256"
    ):
        errors.append("threshold_configuration_sha256 mismatch between generations")
    if run1["decision"].get("thresholds_sha256") != run2["decision"].get("thresholds_sha256"):
        errors.append("thresholds_sha256 mismatch between generations")
    if run1["decision"].get("tokenizer_fingerprint") != run2["decision"].get(
        "tokenizer_fingerprint"
    ):
        errors.append("tokenizer_fingerprint mismatch between generations")

    return errors


def _setup_fixture_copy(
    dest_root: Path,
    report_src: Path,
    decision_src: Path,
    manifest: dict[str, Any],
) -> Path:
    ev_dir = dest_root / "evidence" / "tokenizer" / "milestone-6-1-synthetic"
    ev_dir.mkdir(parents=True)
    shutil.copy2(report_src, ev_dir / "evaluation-report.json")
    shutil.copy2(decision_src, ev_dir / "acceptance-decision.json")
    (ev_dir / "manifest.json").write_bytes(_canonical_bytes(manifest))
    for src_rel in (_TOKENIZER, _DATASET, _THRESHOLDS):
        dst = dest_root / src_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(_REPO_ROOT / src_rel), str(dst))
    return ev_dir / "manifest.json"


def _compare_generated_to_committed(run: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    committed_report = _EVIDENCE_DIR / "evaluation-report.json"
    committed_decision = _EVIDENCE_DIR / "acceptance-decision.json"
    committed_manifest = _EVIDENCE_DIR / "manifest.json"

    _compare_byte("committed vs generated report", committed_report, run["report_path"], errors)
    _compare_byte(
        "committed vs generated decision", committed_decision, run["decision_path"], errors
    )

    committed_manifest_payload = json.loads(committed_manifest.read_text(encoding="utf-8"))
    if _canonical_bytes(committed_manifest_payload) != _canonical_bytes(run["manifest"]):
        errors.append("committed vs generated manifest payload mismatch")

    for keypath, gen_val in [
        (("evaluation_report", "sha256"), _digest(run["report_path"])),
        (("evaluation_report", "report_sha256"), run["report"].get("report_sha256", "")),
        (("acceptance_decision", "sha256"), _digest(run["decision_path"])),
        (
            ("acceptance_decision", "acceptance_sha256"),
            run["decision"].get("acceptance_sha256", ""),
        ),
        (
            ("acceptance_decision", "input_report_sha256"),
            run["decision"].get("input_report_sha256", ""),
        ),
        (("acceptance_decision", "tokenizer_name"), _TOKENIZER_NAME),
        (
            ("acceptance_decision", "tokenizer_fingerprint"),
            run["decision"].get("tokenizer_fingerprint", ""),
        ),
        (("acceptance_decision", "passed"), run["decision"].get("passed", False)),
    ]:
        committed_val: Any = committed_manifest_payload
        for key in keypath:
            committed_val = (
                committed_val.get(key, "__missing__")
                if isinstance(committed_val, dict)
                else "__missing__"
            )
        if committed_val != gen_val:
            errors.append(
                f"committed vs generated manifest {'.'.join(keypath)}: "
                f"committed={committed_val!r}, generated={gen_val!r}"
            )

    committed_validation_errors = validate_evidence(committed_manifest)
    for err in committed_validation_errors:
        errors.append(f"committed evidence validation: {err}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic regeneration of synthetic tokenizer evidence"
    )
    parser.add_argument(
        "--verify-committed",
        action="store_true",
        help="Verify generated evidence equals committed evidence",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        run1 = _generate(tmp / "run1")
        run2 = _generate(tmp / "run2")

        errors = _compare_generations(run1, run2)
        if errors:
            for err in errors:
                print(f"error: {err}", file=sys.stderr)
            return 1

        print("Generation is deterministic: byte-identical across two runs")

        if args.verify_committed:
            generated_manifest = _setup_fixture_copy(
                tmp / "verify",
                run1["report_path"],
                run1["decision_path"],
                run1["manifest"],
            )
            generated_validation_errors = validate_evidence(generated_manifest)
            if generated_validation_errors:
                for err in generated_validation_errors:
                    print(f"error: generated evidence validation: {err}", file=sys.stderr)
                return 1

            committed_errors = _compare_generated_to_committed(run1)
            if committed_errors:
                for err in committed_errors:
                    print(f"error: {err}", file=sys.stderr)
                return 1

            print("Generated evidence matches committed evidence")
            print("Committed evidence pack passes full provenance validation")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
