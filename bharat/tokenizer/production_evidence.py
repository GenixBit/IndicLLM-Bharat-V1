from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from bharat.tokenizer.acceptance import ThresholdConfiguration
from bharat.tokenizer.evaluation import validate_evaluation_report
from bharat.tokenizer.loader import load_tokenizer

_SCHEMA_VERSION = "tokenizer-production-evidence-manifest-v1"
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = {
    "schema_version",
    "status",
    "evidence_scope",
    "repository_commit_sha",
    "tokenizer",
    "evaluation_input",
    "evaluation_report",
    "acceptance_decision",
    "threshold_configuration",
    "language_coverage",
    "generating_commands",
}


@dataclass(frozen=True)
class ProductionEvidenceValidation:
    manifest_sha256: str
    status: str
    accepted: bool
    errors: tuple[str, ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tokenizer-production-evidence-validation-v1",
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "accepted": self.accepted,
            "errors": list(self.errors),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_canonical_dict())


def _reject_non_finite(value: str) -> NoReturn:
    raise ValueError(f"JSON contains non-finite value: {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_canonical_json(path: Path, errors: list[str], label: str) -> Any | None:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_finite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label}: invalid JSON: {exc}")
        return None
    if raw != _canonical_bytes(value):
        errors.append(f"{label}: JSON bytes are not canonical")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_file(root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: path must be a non-empty string")
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        errors.append(f"{label}: path must be relative")
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label}: path escapes evidence root")
        return None
    if not resolved.is_file():
        errors.append(f"{label}: file not found")
        return None
    return resolved


def _file_reference(
    root: Path,
    value: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        errors.append(f"{label}: expected exactly path and sha256")
        return None
    digest = value.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        errors.append(f"{label}.sha256: invalid SHA-256")
    path = _resolve_file(root, value.get("path"), f"{label}.path", errors)
    if path is not None and digest != _sha256(path):
        errors.append(f"{label}: SHA-256 mismatch")
    return path


def _validate_manifest_shape(manifest: Any, errors: list[str]) -> bool:
    if not isinstance(manifest, dict):
        errors.append("manifest must be an object")
        return False
    missing = sorted(_REQUIRED_KEYS - set(manifest))
    unknown = sorted(set(manifest) - _REQUIRED_KEYS)
    if missing:
        errors.append(f"manifest missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"manifest unknown keys: {', '.join(unknown)}")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        errors.append("unsupported manifest schema_version")
    if manifest.get("status") not in {"candidate", "accepted"}:
        errors.append("manifest status must be candidate or accepted")
    if manifest.get("evidence_scope") != "production-local-approved":
        errors.append("manifest evidence_scope must be production-local-approved")
    commit = manifest.get("repository_commit_sha")
    if not isinstance(commit, str) or _GIT_OBJECT_ID.fullmatch(commit) is None:
        errors.append(
            "repository_commit_sha must be a lowercase 40- or 64-character Git object ID"
        )
    commands = manifest.get("generating_commands")
    if not isinstance(commands, list) or not commands or not all(
        isinstance(command, str) and command for command in commands
    ):
        errors.append("generating_commands must be a non-empty string array")
    return not missing and not unknown


def validate_production_evidence(manifest_path: Path) -> ProductionEvidenceValidation:
    errors: list[str] = []
    manifest = _load_canonical_json(manifest_path, errors, "manifest")
    manifest_digest = _sha256(manifest_path) if manifest_path.is_file() else "0" * 64
    status = (
        manifest.get("status", "invalid") if isinstance(manifest, dict) else "invalid"
    )
    if not _validate_manifest_shape(manifest, errors):
        return ProductionEvidenceValidation(
            manifest_digest,
            str(status),
            False,
            tuple(errors),
        )

    assert isinstance(manifest, dict)
    root = manifest_path.parent.resolve()
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append("tokenizer must be an object")
        tokenizer = {}
    expected_tokenizer_keys = {
        "artifact_path",
        "artifact_sha256",
        "fingerprint",
        "vocab_size",
        "normalization",
        "byte_alphabet_complete",
    }
    if set(tokenizer) != expected_tokenizer_keys:
        errors.append("tokenizer has missing or unknown keys")
    artifact_path = _resolve_file(
        root,
        tokenizer.get("artifact_path"),
        "tokenizer.artifact_path",
        errors,
    )
    artifact_digest = tokenizer.get("artifact_sha256")
    if (
        not isinstance(artifact_digest, str)
        or _SHA256.fullmatch(artifact_digest) is None
    ):
        errors.append("tokenizer.artifact_sha256: invalid SHA-256")
    elif artifact_path is not None and artifact_digest != _sha256(artifact_path):
        errors.append("tokenizer artifact SHA-256 mismatch")

    loaded_tokenizer = None
    if artifact_path is not None:
        try:
            loaded_tokenizer = load_tokenizer(str(artifact_path))
            if loaded_tokenizer.fingerprint() != tokenizer.get("fingerprint"):
                errors.append("tokenizer fingerprint mismatch")
            if loaded_tokenizer.vocab_size != tokenizer.get("vocab_size"):
                errors.append("tokenizer vocab_size mismatch")
            if loaded_tokenizer.get_metadata().get("normalization") != "NFC":
                errors.append("tokenizer normalization is not NFC")
        except Exception as exc:
            errors.append(f"tokenizer cannot be validated: {exc}")

    input_path = _file_reference(
        root,
        manifest.get("evaluation_input"),
        "evaluation_input",
        errors,
    )
    report_path = _file_reference(
        root,
        manifest.get("evaluation_report"),
        "evaluation_report",
        errors,
    )
    decision_path = _file_reference(
        root,
        manifest.get("acceptance_decision"),
        "acceptance_decision",
        errors,
    )
    thresholds_path = _file_reference(
        root,
        manifest.get("threshold_configuration"),
        "threshold_configuration",
        errors,
    )

    report: Any | None = None
    if report_path is not None:
        report = _load_canonical_json(report_path, errors, "evaluation_report")
        if isinstance(report, dict):
            try:
                validate_evaluation_report(report)
            except Exception as exc:
                errors.append(f"evaluation_report: {exc}")

    decision = (
        _load_canonical_json(decision_path, errors, "acceptance_decision")
        if decision_path is not None
        else None
    )
    threshold_configuration = None
    if thresholds_path is not None:
        payload = _load_canonical_json(
            thresholds_path,
            errors,
            "threshold_configuration",
        )
        if isinstance(payload, dict):
            try:
                threshold_configuration = ThresholdConfiguration.from_payload(payload)
            except Exception as exc:
                errors.append(f"threshold_configuration: {exc}")

    coverage = manifest.get("language_coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "required_languages",
        "record_counts",
    }:
        errors.append(
            "language_coverage must contain exactly required_languages and record_counts"
        )
    else:
        required = coverage.get("required_languages")
        counts = coverage.get("record_counts")
        if (
            not isinstance(required, list)
            or not required
            or len(required) != len(set(required))
        ):
            errors.append("required_languages must be a non-empty unique array")
        if not isinstance(counts, dict):
            errors.append("record_counts must be an object")
        elif isinstance(required, list):
            for language in required:
                if not isinstance(language, str) or language not in counts:
                    errors.append(
                        f"required language missing from record_counts: {language!r}"
                    )
                elif not isinstance(counts[language], int) or isinstance(
                    counts[language], bool
                ):
                    errors.append(f"record count for {language!r} must be an integer")

    if status == "accepted":
        if tokenizer.get("byte_alphabet_complete") is not True:
            errors.append("accepted evidence requires complete byte coverage")
        if not isinstance(decision, dict) or decision.get("passed") is not True:
            errors.append("accepted evidence requires a passing acceptance decision")
        if (
            threshold_configuration is None
            or threshold_configuration.status != "production"
        ):
            errors.append("accepted evidence requires production thresholds")
        if (
            threshold_configuration is not None
            and threshold_configuration.evidence_scope != "approved-evaluation-set"
        ):
            errors.append("accepted evidence requires approved-evaluation-set thresholds")
        if (
            isinstance(decision, dict)
            and isinstance(report, dict)
            and decision.get("input_report_sha256") != report.get("report_sha256")
        ):
            errors.append("acceptance decision does not reference the evaluation report")
        if input_path is None or loaded_tokenizer is None:
            errors.append("accepted evidence is not independently verifiable")

    return ProductionEvidenceValidation(
        manifest_digest,
        str(status),
        status == "accepted" and not errors,
        tuple(errors),
    )
