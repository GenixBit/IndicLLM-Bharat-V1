from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import ThresholdConfiguration, evaluate_tokenizer_acceptance
from bharat.tokenizer.evaluation import validate_evaluation_report


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(obj: dict[str, Any], exclude: str | None = None) -> str:
    payload = {k: v for k, v in obj.items() if k != exclude}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check(condition: bool, msg: str, errors: list[str]) -> None:
    if not condition:
        errors.append(msg)


def validate_evidence(manifest_path: Path) -> list[str]:
    errors: list[str] = []

    if not manifest_path.is_file():
        return [f"manifest not found: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    manifest_dir = manifest_path.parent
    # Resolve shared artifact paths relative to repository root
    repo_root = manifest_path.parents[3]
    # Resolve generated evidence paths relative to manifest directory
    _evidence_dir = manifest_dir

    # ── Schema version ────────────────────────────────────────────
    _check(
        manifest.get("schema_version") == "tokenizer-evidence-manifest-v1",
        "unsupported manifest schema_version",
        errors,
    )

    # ── Status ────────────────────────────────────────────────────
    _check(manifest.get("status") == "provisional", "status must be provisional", errors)

    # ── Evidence scope ────────────────────────────────────────────
    _check(
        manifest.get("evidence_scope") == "synthetic-local-only",
        "evidence_scope must be synthetic-local-only",
        errors,
    )

    # ── Tokenizer artifact ────────────────────────────────────────
    tok = manifest.get("tokenizer", {})
    if not isinstance(tok, dict):
        errors.append("tokenizer entry must be an object")
    else:
        tok_path = Path(str(tok.get("artifact_path", "")))
        if tok_path.is_absolute():
            errors.append(f"tokenizer artifact_path must be relative: {tok_path}")
        else:
            resolved = (repo_root / tok_path).resolve()
            if not resolved.is_file():
                errors.append(f"tokenizer artifact not found: {resolved}")
            else:
                actual = _digest(resolved)
                expected = str(tok.get("artifact_sha256", ""))
                _check(actual == expected, f"tokenizer artifact SHA-256 mismatch", errors)

        if tok.get("fingerprint"):
            try:
                from bharat.tokenizer import load_tokenizer

                t = load_tokenizer(str((repo_root / Path(str(tok.get("artifact_path", "")))).resolve()))
                actual_fp = t.fingerprint()
                _check(
                    actual_fp == tok["fingerprint"],
                    f"tokenizer fingerprint mismatch: expected {tok['fingerprint']}, got {actual_fp}",
                    errors,
                )
            except Exception as exc:
                errors.append(f"cannot load tokenizer: {exc}")

    # ── Evaluation fixture ────────────────────────────────────────
    fixture = manifest.get("evaluation_fixture", {})
    if not isinstance(fixture, dict):
        errors.append("evaluation_fixture entry must be an object")
    else:
        fixture_path = Path(str(fixture.get("path", "")))
        if fixture_path.is_absolute():
            errors.append(f"evaluation_fixture path must be relative: {fixture_path}")
        else:
            resolved = (repo_root / fixture_path).resolve()
            if not resolved.is_file():
                errors.append(f"evaluation fixture not found: {resolved}")
            else:
                actual = _digest(resolved)
                expected = str(fixture.get("sha256", ""))
                _check(actual == expected, f"evaluation fixture SHA-256 mismatch", errors)

    # ── Threshold configuration ───────────────────────────────────
    tc = manifest.get("threshold_configuration", {})
    if not isinstance(tc, dict):
        errors.append("threshold_configuration entry must be an object")
    else:
        tc_path = Path(str(tc.get("path", "")))
        if tc_path.is_absolute():
            errors.append(f"threshold_configuration path must be relative: {tc_path}")
        else:
            resolved = (repo_root / tc_path).resolve()
            if not resolved.is_file():
                errors.append(f"threshold configuration not found: {resolved}")
            else:
                actual = _digest(resolved)
                expected = str(tc.get("sha256", ""))
                _check(actual == expected, f"threshold configuration SHA-256 mismatch", errors)

                try:
                    config = ThresholdConfiguration.from_payload(json.loads(resolved.read_text(encoding="utf-8")))
                    actual_ts = _canonical_digest(config.thresholds.to_canonical_dict())
                    _check(
                        actual_ts == str(tc.get("thresholds_sha256", "")),
                        "thresholds SHA-256 mismatch",
                        errors,
                    )
                    actual_cs = config.configuration_sha256()
                    _check(
                        actual_cs == str(tc.get("configuration_sha256", "")),
                        "configuration SHA-256 mismatch",
                        errors,
                    )
                except Exception as exc:
                    errors.append(f"cannot validate threshold configuration: {exc}")

    # ── Evaluation report ─────────────────────────────────────────
    er = manifest.get("evaluation_report", {})
    if not isinstance(er, dict):
        errors.append("evaluation_report entry must be an object")
    else:
        er_path_str = str(er.get("path", ""))
        er_path = Path(er_path_str)
        if er_path.is_absolute():
            errors.append(f"evaluation_report path must be relative: {er_path}")
        else:
            er_resolved_base = manifest_dir if not er_path_str.startswith("tests/") else repo_root
            resolved = (er_resolved_base / er_path).resolve()
            if not resolved.is_file():
                errors.append(f"evaluation report not found: {resolved}")
            else:
                actual = _digest(resolved)
                expected = str(er.get("sha256", ""))
                _check(actual == expected, f"evaluation report SHA-256 mismatch", errors)

                try:
                    report = json.loads(resolved.read_text(encoding="utf-8"))
                    validate_evaluation_report(report)

                    actual_report_digest = _canonical_digest(report, "report_sha256")
                    _check(
                        actual_report_digest == report.get("report_sha256", ""),
                        "evaluation report internal digest mismatch",
                        errors,
                    )

                    actual_ds = report.get("input_dataset_sha256", "")
                    expected_ds = str(er.get("input_dataset_sha256", ""))
                    _check(
                        actual_ds == expected_ds,
                        f"input_dataset_sha256 mismatch: expected {expected_ds}, got {actual_ds}",
                        errors,
                    )
                except Exception as exc:
                    errors.append(f"cannot validate evaluation report: {exc}")

    # ── Acceptance decision ───────────────────────────────────────
    ad = manifest.get("acceptance_decision", {})
    if not isinstance(ad, dict):
        errors.append("acceptance_decision entry must be an object")
    else:
        ad_path_str = str(ad.get("path", ""))
        ad_path = Path(ad_path_str)
        if ad_path.is_absolute():
            errors.append(f"acceptance_decision path must be relative: {ad_path}")
        else:
            ad_resolved_base = manifest_dir if not ad_path_str.startswith("tests/") else repo_root
            resolved = (ad_resolved_base / ad_path).resolve()
            if not resolved.is_file():
                errors.append(f"acceptance decision not found: {resolved}")
            else:
                actual = _digest(resolved)
                expected = str(ad.get("sha256", ""))
                _check(actual == expected, f"acceptance decision SHA-256 mismatch", errors)

                try:
                    decision = json.loads(resolved.read_text(encoding="utf-8"))
                    reported_pass = decision.get("passed", False)
                    manifest_pass = ad.get("passed", False)
                    _check(
                        reported_pass == manifest_pass,
                        f"acceptance passed flag mismatch: manifest={manifest_pass}, decision={reported_pass}",
                        errors,
                    )
                    _check(
                        decision.get("threshold_configuration_status") == "provisional",
                        "acceptance decision must have provisional status",
                        errors,
                    )
                    _check(
                        decision.get("threshold_evidence_scope") == "synthetic-local-only",
                        "acceptance decision must have synthetic-local-only evidence_scope",
                        errors,
                    )

                    # Recompute acceptance digest from file
                    actual_ad_digest = _canonical_digest(decision, "acceptance_sha256")
                    _check(
                        actual_ad_digest == decision.get("acceptance_sha256", ""),
                        "acceptance decision internal digest mismatch",
                        errors,
                    )
                except Exception as exc:
                    errors.append(f"cannot validate acceptance decision: {exc}")

    # ── No timestamps, absolute paths, or production claims ───────
    manifest_str = json.dumps(manifest)
    if any(
        keyword in manifest_str
        for keyword in [
            "timestamp",
            "datetime",
            "created_at",
            "updated_at",
            "production",
            "approved-evaluation-set",
        ]
    ):
        errors.append("manifest must not contain timestamp or production fields")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a tokenizer evidence pack manifest"
    )
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    args = parser.parse_args(argv)

    errors = validate_evidence(args.manifest)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    print("evidence pack is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
