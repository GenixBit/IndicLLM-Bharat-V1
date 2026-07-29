from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import (
    ThresholdConfiguration,
    evaluate_tokenizer_acceptance,
)
from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.evaluation import (
    compute_evaluation_dataset_sha256,
    load_evaluation_records,
    validate_evaluation_report,
)
from bharat.tokenizer.production_evidence import (
    byte_alphabet_complete,
    reject_non_finite,
    validate_production_evidence,
)

_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=reject_non_finite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}: top-level value must be a JSON object")
    if raw != _canonical_bytes(value):
        raise ValueError(f"{label}: JSON bytes are not canonical")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_file(root: Path, path: Path, label: str) -> tuple[Path, str]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside evidence root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist or is not a file")
    if path.is_symlink() and path.parent.resolve() != resolved_root:
        raise ValueError(f"{label} symlink parent must remain inside evidence root")
    return resolved, relative.as_posix()


def _validate_manifest_root(
    output_path: Path,
    evidence_root: Path,
) -> None:
    out = output_path.resolve()
    root = evidence_root.resolve()
    if not root.is_dir():
        raise ValueError("evidence_root must be an existing directory")
    if out.parent != root:
        raise ValueError(
            f"output must be directly inside evidence_root, "
            f"got parent={out.parent}, expected={root}"
        )
    if out == root:
        raise ValueError("output must be a file, not evidence_root itself")


def _check_output_path(output_path: Path) -> None:
    out = output_path.resolve()
    if out.exists() and out.is_file():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")


def build_candidate_manifest(
    *,
    evidence_root: Path,
    repository_commit_sha: str,
    tokenizer_path: Path,
    evaluation_input_path: Path,
    evaluation_report_path: Path,
    acceptance_decision_path: Path,
    threshold_configuration_path: Path,
    generating_commands: list[str],
) -> dict[str, Any]:
    """Build a deterministic candidate manifest from caller-provided local evidence."""

    if _GIT_OBJECT_ID.fullmatch(repository_commit_sha) is None:
        raise ValueError("repository_commit_sha must be a lowercase 40- or 64-character hex ID")
    if not generating_commands or any(
        not isinstance(item, str) or not item.strip() for item in generating_commands
    ):
        raise ValueError("generating_commands must contain non-empty strings")

    root = evidence_root.resolve()
    if not root.is_dir():
        raise ValueError("evidence_root must be an existing directory")

    tokenizer_file, tokenizer_relative = _relative_file(root, tokenizer_path, "tokenizer_path")
    input_file, input_relative = _relative_file(
        root, evaluation_input_path, "evaluation_input_path"
    )
    report_file, report_relative = _relative_file(
        root, evaluation_report_path, "evaluation_report_path"
    )
    decision_file, decision_relative = _relative_file(
        root, acceptance_decision_path, "acceptance_decision_path"
    )
    thresholds_file, thresholds_relative = _relative_file(
        root, threshold_configuration_path, "threshold_configuration_path"
    )

    report = _load_canonical_json(report_file, "evaluation_report")
    decision = _load_canonical_json(decision_file, "acceptance_decision")
    thresholds_payload = _load_canonical_json(thresholds_file, "threshold_configuration")

    validate_evaluation_report(report)

    threshold_config = ThresholdConfiguration.from_payload(thresholds_payload)

    tokenizer_name = decision.get("tokenizer_name")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError("acceptance_decision tokenizer_name must be a non-empty string")

    recomputed = evaluate_tokenizer_acceptance(report, tokenizer_name, threshold_config)
    if decision != recomputed:
        raise ValueError("acceptance_decision does not match recomputed decision")

    loaded = BPETokenizer.load(tokenizer_file)
    if not byte_alphabet_complete(loaded):
        raise ValueError("tokenizer artifact does not contain a complete byte alphabet")

    tokenizer_fp = loaded.compute_hash()

    report_fp_map = report.get("tokenizer_fingerprints")
    if not isinstance(report_fp_map, dict) or report_fp_map.get(tokenizer_name) != tokenizer_fp:
        raise ValueError("evaluation_report tokenizer_fingerprint does not match loaded tokenizer")

    decision_fp = decision.get("tokenizer_fingerprint")
    if decision_fp != tokenizer_fp:
        raise ValueError(
            "acceptance_decision tokenizer_fingerprint does not match loaded tokenizer"
        )

    records = load_evaluation_records(input_file)
    dataset_digest = compute_evaluation_dataset_sha256(records)
    report_ds = report.get("input_dataset_sha256")
    if report_ds != dataset_digest:
        raise ValueError(
            f"evaluation_input dataset digest {dataset_digest} does not match "
            f"report input_dataset_sha256 {report_ds}"
        )

    per_language = report.get("per_language")
    if not isinstance(per_language, dict):
        raise ValueError("evaluation_report per_language must be an object")
    tokenizer_languages = per_language.get(tokenizer_name)
    if not isinstance(tokenizer_languages, dict) or not tokenizer_languages:
        raise ValueError("evaluation_report must contain per-language metrics for tokenizer_name")

    record_counts: dict[str, int] = {}
    for language, metrics in tokenizer_languages.items():
        if not isinstance(language, str) or not isinstance(metrics, dict):
            raise ValueError("evaluation_report per-language entries are invalid")
        count = metrics.get("record_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"evaluation_report record_count for {language!r} must be positive")
        lang_records = [r for r in records if r.language == language]
        if len(lang_records) != count:
            raise ValueError(
                f"evaluation_input has {len(lang_records)} records for language "
                f"{language!r} but report claims {count}"
            )
        record_counts[language] = count

    return {
        "schema_version": "tokenizer-production-evidence-manifest-v1",
        "status": "candidate",
        "evidence_scope": "production-local-approved",
        "repository_commit_sha": repository_commit_sha,
        "tokenizer": {
            "artifact_path": tokenizer_relative,
            "artifact_sha256": _sha256(tokenizer_file),
            "fingerprint": tokenizer_fp,
            "vocab_size": loaded.vocab_size,
            "normalization": "NFC",
            "byte_alphabet_complete": True,
        },
        "evaluation_input": {
            "path": input_relative,
            "sha256": _sha256(input_file),
        },
        "evaluation_report": {
            "path": report_relative,
            "sha256": _sha256(report_file),
        },
        "acceptance_decision": {
            "path": decision_relative,
            "sha256": _sha256(decision_file),
        },
        "threshold_configuration": {
            "path": thresholds_relative,
            "sha256": _sha256(thresholds_file),
        },
        "language_coverage": {
            "required_languages": sorted(record_counts),
            "record_counts": {key: record_counts[key] for key in sorted(record_counts)},
        },
        "generating_commands": list(generating_commands),
    }


def _publish_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    reread = path.read_bytes()
    if reread != payload:
        raise RuntimeError(
            f"byte-verification failed for {path}: "
            f"read-back {len(reread)} bytes, expected {len(payload)}"
        )


def write_candidate_manifest(output_path: Path, **kwargs: Any) -> str:
    """Validate and publish a canonical candidate manifest without overwriting files."""

    _check_output_path(output_path)

    evidence_root = kwargs.get("evidence_root")
    if evidence_root is not None:
        _validate_manifest_root(output_path, evidence_root)

    manifest = build_candidate_manifest(**kwargs)
    payload = _canonical_bytes(manifest)

    output = output_path.resolve()
    final_digest = hashlib.sha256(payload).hexdigest()

    temp = output.with_name(f".{output.name}.{secrets.token_hex(8)}.tmp")
    created: list[Path] = []
    try:
        _publish_exclusive(temp, payload)

        created.append(temp)

        validation = validate_production_evidence(temp)
        if not validation.valid:
            raise ValueError(
                "candidate evidence validation failed: " + "; ".join(validation.errors)
            )

        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

        _publish_exclusive(output, payload)
        created.append(output)

        output_recheck = hashlib.sha256(output.read_bytes()).hexdigest()
        if output_recheck != final_digest:
            raise RuntimeError(
                f"final SHA-256 mismatch: computed {output_recheck}, expected {final_digest}"
            )
    except BaseException:
        for f in created:
            f.unlink(missing_ok=True)
        raise
    finally:
        for f in created:
            if f != output:
                f.unlink(missing_ok=True)

    return final_digest
