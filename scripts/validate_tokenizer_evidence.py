from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

from bharat.tokenizer.acceptance import (
    ThresholdConfiguration,
    evaluate_tokenizer_acceptance,
)
from bharat.tokenizer.evaluation import validate_evaluation_report


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(obj: dict[str, Any], exclude: str | None = None) -> str:
    payload = {k: v for k, v in obj.items() if k != exclude}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check(condition: bool, msg: str, errors: list[str]) -> None:
    if not condition:
        errors.append(msg)


def _sha256_re(s: str) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _reject_non_finite(value: str) -> NoReturn:
    raise ValueError(f"JSON contains non-finite value: {value!r}")


def _load_json_strict(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_reject_non_finite)


def _resolve_path(
    path_str: str,
    base: Path,
    *,
    label: str,
    errors: list[str],
) -> Path | None:
    if not path_str:
        errors.append(f"{label}: path is empty")
        return None
    p = Path(path_str)
    if p.is_absolute():
        errors.append(f"{label}: path must be relative: {path_str}")
        return None
    resolved = (base / p).resolve()
    base_resolved = base.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        errors.append(f"{label}: path escapes base directory: {path_str}")
        return None
    if resolved.is_symlink():
        target = resolved.readlink()
        if not target.is_absolute():
            target = (resolved.parent / target).resolve()
        else:
            target = target.resolve()
        try:
            target.relative_to(base_resolved)
        except ValueError:
            errors.append(f"{label}: symlink target escapes base directory: {path_str} -> {target}")
            return None
    if not resolved.is_file():
        errors.append(f"{label}: not found: {resolved}")
        return None
    return resolved


def _validate_manifest_section(
    section_name: str,
    section: Any,
    allowed_keys: set[str],
    required_keys: set[str],
    key_types: dict[str, type],
    sha256_keys: set[str],
    errors: list[str],
) -> bool:
    if not isinstance(section, dict):
        errors.append(f"{section_name}: must be an object")
        return False
    unknown = sorted(set(section) - allowed_keys)
    if unknown:
        errors.append(f"{section_name}: unknown keys: {', '.join(unknown)}")
        return False
    for key in required_keys:
        if key not in section:
            errors.append(f"{section_name}: missing required key: {key!r}")
            return False
    for key, expected_type in key_types.items():
        if key in section:
            if expected_type is bool:
                if not isinstance(section[key], bool):
                    errors.append(
                        f"{section_name}.{key}: expected {expected_type.__name__}, "
                        f"got {type(section[key]).__name__}"
                    )
            elif expected_type is str:
                if not isinstance(section[key], str):
                    errors.append(
                        f"{section_name}.{key}: expected {expected_type.__name__}, "
                        f"got {type(section[key]).__name__}"
                    )
            elif expected_type is int:
                if not isinstance(section[key], int) or isinstance(section[key], bool):
                    errors.append(
                        f"{section_name}.{key}: expected {expected_type.__name__}, "
                        f"got {type(section[key]).__name__}"
                    )
            elif expected_type is float and (
                not isinstance(section[key], int | float) or isinstance(section[key], bool)
            ):
                errors.append(
                    f"{section_name}.{key}: expected number, " f"got {type(section[key]).__name__}"
                )
    for key in sha256_keys:
        if key in section and not _sha256_re(str(section[key])):
            errors.append(f"{section_name}.{key}: not a valid SHA-256 hex string")
    return True


def validate_evidence(manifest_path: Path) -> list[str]:
    errors: list[str] = []

    if not manifest_path.is_file():
        return [f"manifest not found: {manifest_path}"]

    try:
        manifest = _load_json_strict(manifest_path)
    except (json.JSONDecodeError, ValueError) as exc:
        return [f"manifest JSON error: {exc}"]

    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    manifest_dir = manifest_path.parent
    repo_root = manifest_path.parents[3]

    # ── Manifest top-level schema ─────────────────────────────────
    allowed_manifest_keys = {
        "schema_version",
        "status",
        "evidence_scope",
        "tokenizer",
        "evaluation_fixture",
        "threshold_configuration",
        "evaluation_report",
        "acceptance_decision",
        "generating_commands",
    }
    unknown_mk = sorted(set(manifest) - allowed_manifest_keys)
    if unknown_mk:
        errors.append(f"manifest: unknown keys: {', '.join(unknown_mk)}")

    _check(
        manifest.get("schema_version") == "tokenizer-evidence-manifest-v1",
        "unsupported manifest schema_version",
        errors,
    )
    _check(
        manifest.get("status") == "provisional",
        "status must be provisional",
        errors,
    )
    _check(
        manifest.get("evidence_scope") == "synthetic-local-only",
        "evidence_scope must be synthetic-local-only",
        errors,
    )

    # ── Tokenizer artifact ────────────────────────────────────────
    tok = manifest.get("tokenizer", {})
    allowed_tok_keys = {"artifact_path", "artifact_sha256", "fingerprint"}
    required_tok_keys = {"artifact_path", "artifact_sha256"}
    tok_key_types: dict[str, type] = {
        "artifact_path": str,
        "artifact_sha256": str,
        "fingerprint": str,
    }
    if _validate_manifest_section(
        "tokenizer",
        tok,
        allowed_tok_keys,
        required_tok_keys,
        tok_key_types,
        {"artifact_sha256", "fingerprint"},
        errors,
    ):
        tok_path = _resolve_path(
            tok["artifact_path"], repo_root, label="tokenizer.artifact_path", errors=errors
        )
        if tok_path is not None:
            actual = _digest(tok_path)
            _check(
                actual == tok.get("artifact_sha256", ""),
                "tokenizer artifact SHA-256 mismatch",
                errors,
            )
            if tok.get("fingerprint"):
                try:
                    from bharat.tokenizer import load_tokenizer

                    t = load_tokenizer(str(tok_path))
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
    allowed_fix_keys = {"path", "sha256"}
    required_fix_keys = {"path", "sha256"}
    fix_key_types: dict[str, type] = {"path": str, "sha256": str}
    if _validate_manifest_section(
        "evaluation_fixture",
        fixture,
        allowed_fix_keys,
        required_fix_keys,
        fix_key_types,
        {"sha256"},
        errors,
    ):
        fixture_path = _resolve_path(
            fixture["path"], repo_root, label="evaluation_fixture.path", errors=errors
        )
        if fixture_path is not None:
            actual = _digest(fixture_path)
            _check(
                actual == fixture.get("sha256", ""),
                "evaluation fixture SHA-256 mismatch",
                errors,
            )

    # ── Threshold configuration ───────────────────────────────────
    tc = manifest.get("threshold_configuration", {})
    allowed_tc_keys = {
        "path",
        "sha256",
        "thresholds_sha256",
        "configuration_sha256",
    }
    required_tc_keys = {"path", "sha256", "thresholds_sha256", "configuration_sha256"}
    tc_key_types: dict[str, type] = {
        "path": str,
        "sha256": str,
        "thresholds_sha256": str,
        "configuration_sha256": str,
    }
    tc_config: ThresholdConfiguration | None = None
    if _validate_manifest_section(
        "threshold_configuration",
        tc,
        allowed_tc_keys,
        required_tc_keys,
        tc_key_types,
        {"sha256", "thresholds_sha256", "configuration_sha256"},
        errors,
    ):
        tc_path = _resolve_path(
            tc["path"], repo_root, label="threshold_configuration.path", errors=errors
        )
        if tc_path is not None:
            actual = _digest(tc_path)
            _check(
                actual == tc.get("sha256", ""),
                "threshold configuration SHA-256 mismatch",
                errors,
            )
            try:
                tc_payload = json.loads(tc_path.read_text(encoding="utf-8"))
                tc_config = ThresholdConfiguration.from_payload(tc_payload)
                actual_ts = _canonical_digest(tc_config.thresholds.to_canonical_dict())
                _check(
                    actual_ts == tc.get("thresholds_sha256", ""),
                    "thresholds SHA-256 mismatch",
                    errors,
                )
                actual_cs = tc_config.configuration_sha256()
                _check(
                    actual_cs == tc.get("configuration_sha256", ""),
                    "configuration SHA-256 mismatch",
                    errors,
                )
            except Exception as exc:
                errors.append(f"cannot validate threshold configuration: {exc}")

    # ── Evaluation report ─────────────────────────────────────────
    er = manifest.get("evaluation_report", {})
    allowed_er_keys = {
        "path",
        "sha256",
        "report_sha256",
        "input_dataset_sha256",
    }
    required_er_keys = {"path", "sha256", "report_sha256", "input_dataset_sha256"}
    er_key_types: dict[str, type] = {
        "path": str,
        "sha256": str,
        "report_sha256": str,
        "input_dataset_sha256": str,
    }
    report_data: dict[str, Any] | None = None
    if _validate_manifest_section(
        "evaluation_report",
        er,
        allowed_er_keys,
        required_er_keys,
        er_key_types,
        {"sha256", "report_sha256", "input_dataset_sha256"},
        errors,
    ):
        er_path = _resolve_path(
            er["path"], manifest_dir, label="evaluation_report.path", errors=errors
        )
        if er_path is not None:
            actual = _digest(er_path)
            _check(
                actual == er.get("sha256", ""),
                "evaluation report SHA-256 mismatch",
                errors,
            )
            try:
                report_data = _load_json_strict(er_path)
                validate_evaluation_report(report_data)

                actual_report_digest = _canonical_digest(report_data, "report_sha256")
                _check(
                    actual_report_digest == report_data.get("report_sha256", ""),
                    "evaluation report internal digest mismatch",
                    errors,
                )

                actual_ds = report_data.get("input_dataset_sha256", "")
                expected_ds = er.get("input_dataset_sha256", "")
                _check(
                    actual_ds == expected_ds,
                    f"input_dataset_sha256 mismatch: expected {expected_ds}, got {actual_ds}",
                    errors,
                )

                actual_rs = report_data.get("report_sha256", "")
                expected_rs = er.get("report_sha256", "")
                _check(
                    actual_rs == expected_rs,
                    f"report_sha256 mismatch: expected {expected_rs}, got {actual_rs}",
                    errors,
                )
            except Exception as exc:
                errors.append(f"cannot validate evaluation report: {exc}")

    # ── Acceptance decision ───────────────────────────────────────
    ad = manifest.get("acceptance_decision", {})
    allowed_ad_keys = {
        "path",
        "sha256",
        "acceptance_sha256",
        "input_report_sha256",
        "tokenizer_name",
        "tokenizer_fingerprint",
        "passed",
    }
    required_ad_keys = {
        "path",
        "sha256",
        "acceptance_sha256",
        "input_report_sha256",
        "tokenizer_name",
        "tokenizer_fingerprint",
        "passed",
    }
    ad_key_types: dict[str, type] = {
        "path": str,
        "sha256": str,
        "acceptance_sha256": str,
        "input_report_sha256": str,
        "tokenizer_name": str,
        "tokenizer_fingerprint": str,
        "passed": bool,
    }
    decision_data: dict[str, Any] | None = None
    if _validate_manifest_section(
        "acceptance_decision",
        ad,
        allowed_ad_keys,
        required_ad_keys,
        ad_key_types,
        {"sha256", "acceptance_sha256", "input_report_sha256", "tokenizer_fingerprint"},
        errors,
    ):
        ad_path = _resolve_path(
            ad["path"], manifest_dir, label="acceptance_decision.path", errors=errors
        )
        if ad_path is not None:
            actual = _digest(ad_path)
            _check(
                actual == ad.get("sha256", ""),
                "acceptance decision SHA-256 mismatch",
                errors,
            )
            try:
                decision_data = _load_json_strict(ad_path)

                reported_pass = decision_data.get("passed", False)
                manifest_pass = ad.get("passed", False)
                _check(
                    reported_pass == manifest_pass,
                    f"acceptance passed flag mismatch: manifest={manifest_pass}, decision={reported_pass}",
                    errors,
                )
                _check(
                    decision_data.get("threshold_configuration_status") == "provisional",
                    "acceptance decision must have provisional status",
                    errors,
                )
                _check(
                    decision_data.get("threshold_evidence_scope") == "synthetic-local-only",
                    "acceptance decision must have synthetic-local-only evidence_scope",
                    errors,
                )

                actual_ad_digest = _canonical_digest(decision_data, "acceptance_sha256")
                _check(
                    actual_ad_digest == decision_data.get("acceptance_sha256", ""),
                    "acceptance decision internal digest mismatch",
                    errors,
                )

                # ── Manifest ↔ decision field binding ──────────────
                _check(
                    ad.get("acceptance_sha256", "") == decision_data.get("acceptance_sha256", ""),
                    "manifest acceptance_sha256 != decision acceptance_sha256",
                    errors,
                )
                _check(
                    ad.get("input_report_sha256", "")
                    == decision_data.get("input_report_sha256", ""),
                    "manifest input_report_sha256 != decision input_report_sha256",
                    errors,
                )
                _check(
                    ad.get("tokenizer_name", "") == decision_data.get("tokenizer_name", ""),
                    "manifest tokenizer_name != decision tokenizer_name",
                    errors,
                )
                _check(
                    ad.get("tokenizer_fingerprint", "")
                    == decision_data.get("tokenizer_fingerprint", ""),
                    "manifest tokenizer_fingerprint != decision tokenizer_fingerprint",
                    errors,
                )
                _check(
                    ad.get("passed", False) == decision_data.get("passed", False),
                    "manifest passed != decision passed",
                    errors,
                )

                # Cross-field consistency
                _check(
                    ad.get("input_report_sha256", "") == er.get("report_sha256", ""),
                    "acceptance_decision.input_report_sha256 != evaluation_report.report_sha256",
                    errors,
                )
                _check(
                    ad.get("tokenizer_fingerprint", "") == tok.get("fingerprint", ""),
                    "acceptance_decision.tokenizer_fingerprint != tokenizer.fingerprint",
                    errors,
                )
                tns = report_data.get("tokenizer_names", []) if report_data else []
                tn = ad.get("tokenizer_name", "")
                _check(
                    tns.count(tn) == 1,
                    f"tokenizer_name {tn!r} appears {tns.count(tn)} time(s) in "
                    f"evaluation_report.tokenizer_names (expected 1)",
                    errors,
                )

            except Exception as exc:
                errors.append(f"cannot validate acceptance decision: {exc}")

    # ── Generating commands ────────────────────────────────────────
    gcs = manifest.get("generating_commands", [])
    if not isinstance(gcs, list):
        errors.append("generating_commands must be a list")
    else:
        for i, gc in enumerate(gcs):
            if not isinstance(gc, dict):
                errors.append(f"generating_commands[{i}]: must be an object")
            else:
                allowed_gc = {"module", "arguments"}
                unknown_gc = sorted(set(gc) - allowed_gc)
                if unknown_gc:
                    errors.append(
                        f"generating_commands[{i}]: unknown keys: {', '.join(unknown_gc)}"
                    )
                if not isinstance(gc.get("module", ""), str):
                    errors.append(f"generating_commands[{i}].module: must be a string")
                if not isinstance(gc.get("arguments", []), list):
                    errors.append(f"generating_commands[{i}].arguments: must be a list")
                elif not all(isinstance(a, str) for a in gc.get("arguments", [])):
                    errors.append(f"generating_commands[{i}].arguments: all items must be strings")

    # ── No timestamps, absolute paths, or production claims ───────
    manifest_str = json.dumps(manifest, allow_nan=False)
    for keyword in [
        "timestamp",
        "datetime",
        "created_at",
        "updated_at",
        "production",
        "approved-evaluation-set",
    ]:
        if keyword in manifest_str:
            errors.append(f"manifest must not contain {keyword!r}")
            break

    # ── Full provenance recomputation ─────────────────────────────
    if errors:
        return errors

    if report_data is not None and tc_config is not None and decision_data is not None:
        try:
            tokenizer_name = decision_data.get("tokenizer_name", "")
            if not tokenizer_name:
                errors.append("decision has no tokenizer_name")

            expected_decision = evaluate_tokenizer_acceptance(
                report_data,
                tokenizer_name,
                tc_config,
            )

            expected_canonical = json.dumps(expected_decision, sort_keys=True, allow_nan=False)
            actual_canonical = json.dumps(decision_data, sort_keys=True, allow_nan=False)
            if expected_canonical != actual_canonical:
                errors.append("recomputed acceptance decision does not match committed decision")
                for k in sorted(set(expected_decision) | set(decision_data)):
                    ev = expected_decision.get(k)
                    av = decision_data.get(k)
                    if ev != av:
                        errors.append(f"  field {k!r}: expected={ev!r}, got={av!r}")
            else:
                expected_hash = expected_decision.get("acceptance_sha256", "")
                actual_hash = decision_data.get("acceptance_sha256", "")
                _check(
                    expected_hash == actual_hash,
                    f"acceptance_sha256 mismatch: expected {expected_hash}, got {actual_hash}",
                    errors,
                )

                input_rs = decision_data.get("input_report_sha256", "")
                report_rs = report_data.get("report_sha256", "") if report_data else ""
                _check(
                    input_rs == report_rs,
                    f"decision.input_report_sha256 != report.report_sha256: "
                    f"{input_rs} != {report_rs}",
                    errors,
                )

                input_ds = decision_data.get("input_dataset_sha256", "")
                report_ds = report_data.get("input_dataset_sha256", "") if report_data else ""
                _check(
                    input_ds == report_ds,
                    f"decision.input_dataset_sha256 != report.input_dataset_sha256: "
                    f"{input_ds} != {report_ds}",
                    errors,
                )

                fingerprints = report_data.get("tokenizer_fingerprints", {}) if report_data else {}
                expected_fp = fingerprints.get(tokenizer_name, "")
                decision_fp = decision_data.get("tokenizer_fingerprint", "")
                _check(
                    expected_fp == decision_fp,
                    f"decision tokenizer_fingerprint mismatch: "
                    f"expected {expected_fp}, got {decision_fp}",
                    errors,
                )

                expected_cs = tc_config.configuration_sha256()
                actual_cs = decision_data.get("threshold_configuration_sha256", "")
                _check(
                    expected_cs == actual_cs,
                    f"decision threshold_configuration_sha256 mismatch: "
                    f"expected {expected_cs}, got {actual_cs}",
                    errors,
                )

                actual_ts = decision_data.get("thresholds_sha256", "")
                expected_ts = _canonical_digest(tc_config.thresholds.to_canonical_dict())
                _check(
                    expected_ts == actual_ts,
                    f"decision thresholds_sha256 mismatch: "
                    f"expected {expected_ts}, got {actual_ts}",
                    errors,
                )

        except Exception as exc:
            errors.append(f"cannot recompute acceptance decision: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a tokenizer evidence pack manifest")
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
