from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

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
from bharat.tokenizer.loader import load_tokenizer

_SCHEMA_VERSION = "tokenizer-production-evidence-manifest-v1"
_GIT_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
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
_TOKENIZER_EXPECTED_KEYS = {
    "artifact_path",
    "artifact_sha256",
    "fingerprint",
    "vocab_size",
    "normalization",
    "byte_alphabet_complete",
}
_ALLOWED_STATUS_VALUES = {"candidate", "accepted"}
_ACCEPTED_EVIDENCE_SCOPE = "approved-evaluation-set"


@dataclass(frozen=True)
class ProductionEvidenceValidation:
    manifest_sha256: str
    status: str
    valid: bool
    accepted: bool
    errors: tuple[str, ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tokenizer-production-evidence-validation-v1",
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "valid": self.valid,
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


def _check_sha256(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        errors.append(f"{label}: must be a lowercase 64-character SHA-256")
        return False
    return True


def _check_not_bool(value: Any, label: str, errors: list[str]) -> bool:
    if isinstance(value, bool):
        errors.append(f"{label}: must not be a boolean")
        return False
    return True


def _resolve_file(
    root: Path,
    value: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: path must be a non-empty string")
        return None
    _check_path_safety(value, label, errors)
    candidate = Path(value)
    if candidate.is_absolute():
        errors.append(f"{label}: path must be relative")
        return None
    if ".." in candidate.parts:
        errors.append(f"{label}: path must not contain parent traversal")
        return None
    joined = root / candidate
    if joined.is_symlink():
        target = joined.readlink()
        if not target.is_absolute():
            target = (joined.parent / target).resolve()
        else:
            target = target.resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{label}: symlink target escapes evidence root")
            return None
    resolved = joined.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label}: path escapes evidence root")
        return None
    if not resolved.is_file():
        errors.append(f"{label}: file not found")
        return None
    return resolved


def _check_path_safety(path_str: str, label: str, errors: list[str]) -> None:
    p = PureWindowsPath(path_str)
    if p.drive:
        errors.append(f"{label}: Windows drive path is not allowed")
        return
    if path_str.startswith("\\\\"):
        errors.append(f"{label}: UNC path is not allowed")
        return
    if "\\\\" in path_str or "\\.." in path_str or "..\\" in path_str:
        errors.append(f"{label}: backslash traversal is not allowed")
        return
    p2 = PurePosixPath(path_str)
    if ".." in p2.parts:
        errors.append(f"{label}: parent traversal is not allowed")


def _file_reference(
    root: Path,
    value: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, dict):
        errors.append(f"{label}: must be an object")
        return None
    unknown = sorted(set(value) - {"path", "sha256"})
    if unknown:
        errors.append(f"{label}: unknown keys: {', '.join(unknown)}")
    if "path" not in value:
        errors.append(f"{label}: missing path")
    if "sha256" not in value:
        errors.append(f"{label}: missing sha256")
    digest = value.get("sha256")
    if digest is not None:
        _check_sha256(digest, f"{label}.sha256", errors)
    path = _resolve_file(root, value.get("path"), f"{label}.path", errors)
    if path is not None and digest is not None and digest != _sha256(path):
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
    if manifest.get("status") not in _ALLOWED_STATUS_VALUES:
        errors.append("manifest status must be candidate or accepted")
    if manifest.get("evidence_scope") != "production-local-approved":
        errors.append("manifest evidence_scope must be production-local-approved")
    commit = manifest.get("repository_commit_sha")
    if not isinstance(commit, str) or _GIT_OBJECT_ID.fullmatch(commit) is None:
        errors.append("repository_commit_sha must be a lowercase 40- or 64-character Git object ID")
    commands = manifest.get("generating_commands")
    if (
        not isinstance(commands, list)
        or not commands
        or not all(isinstance(command, str) and command for command in commands)
    ):
        errors.append("generating_commands must be a non-empty string array")
    return not missing and not unknown


def _validate_tokenizer_section(
    tokenizer: Any,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(tokenizer, dict):
        errors.append("tokenizer must be an object")
        return {}
    unknown = sorted(set(tokenizer) - _TOKENIZER_EXPECTED_KEYS)
    if unknown:
        errors.append(f"tokenizer unknown keys: {', '.join(unknown)}")
    missing = sorted(_TOKENIZER_EXPECTED_KEYS - set(tokenizer))
    if missing:
        errors.append(f"tokenizer missing keys: {', '.join(missing)}")
    if tokenizer.get("normalization") != "NFC":
        errors.append("tokenizer.normalization must be NFC")
    byte_alpha = tokenizer.get("byte_alphabet_complete")
    if byte_alpha is not None and not isinstance(byte_alpha, bool):
        errors.append("tokenizer.byte_alphabet_complete must be a boolean")
    vocab = tokenizer.get("vocab_size")
    if vocab is not None:
        if isinstance(vocab, bool):
            errors.append("tokenizer.vocab_size must not be a boolean")
        elif not isinstance(vocab, int) or vocab < 1:
            errors.append("tokenizer.vocab_size must be a positive integer")
    _check_sha256(tokenizer.get("fingerprint"), "tokenizer.fingerprint", errors)
    return tokenizer


def _validate_tokenizer_artifact(
    root: Path,
    tokenizer: dict[str, Any],
    errors: list[str],
) -> tuple[Any | None, BPETokenizer | None]:
    artifact_path = _resolve_file(
        root,
        tokenizer.get("artifact_path"),
        "tokenizer.artifact_path",
        errors,
    )
    artifact_digest = tokenizer.get("artifact_sha256")
    _check_sha256(artifact_digest, "tokenizer.artifact_sha256", errors)
    if (
        artifact_path is not None
        and artifact_digest is not None
        and artifact_digest != _sha256(artifact_path)
    ):
        errors.append("tokenizer artifact SHA-256 mismatch")

    loaded_tokenizer = None
    bpe_tokenizer = None
    if artifact_path is not None:
        try:
            raw_bytes = artifact_path.read_bytes()
            raw_text = raw_bytes.decode("utf-8")
            artifact_data = json.loads(raw_text)
            if isinstance(artifact_data, dict) and artifact_data.get("schema_version") == "bpe-v1":
                bpe_tokenizer = BPETokenizer.load(artifact_path)
                loaded_tokenizer = bpe_tokenizer
                expected_fp = bpe_tokenizer.compute_hash()
                loaded_vocab = bpe_tokenizer.vocab_size
            else:
                loaded_tokenizer = load_tokenizer(str(artifact_path))
                expected_fp = loaded_tokenizer.fingerprint()
                loaded_vocab = loaded_tokenizer.vocab_size
                try:
                    bpe_tokenizer = BPETokenizer.load(artifact_path)
                except Exception as exc:
                    errors.append(f"tokenizer byte alphabet cannot be validated: {exc}")
        except Exception as exc:
            errors.append(f"tokenizer cannot be loaded: {exc}")
            return loaded_tokenizer, bpe_tokenizer
        manifest_fp = tokenizer.get("fingerprint")
        if manifest_fp is not None and expected_fp != manifest_fp:
            errors.append("tokenizer fingerprint mismatch")
        manifest_vocab = tokenizer.get("vocab_size")
        if (
            manifest_vocab is not None
            and not isinstance(manifest_vocab, bool)
            and loaded_vocab != manifest_vocab
        ):
            errors.append(
                f"tokenizer vocab_size mismatch: manifest={manifest_vocab}, "
                f"loaded={loaded_vocab}"
            )
    return loaded_tokenizer, bpe_tokenizer


def _byte_alphabet_complete(tokenizer: BPETokenizer | None) -> bool:
    if tokenizer is None:
        return False
    mapping = tokenizer.byte_value_to_id
    if not isinstance(mapping, dict):
        return False
    if set(mapping) != set(range(256)):
        return False
    token_ids = list(mapping.values())
    if len(set(token_ids)) != 256:
        return False
    id_to_bytes = tokenizer.id_to_bytes
    return all(
        id_to_bytes.get(token_id) == bytes([byte_value]) for byte_value, token_id in mapping.items()
    )


def _validate_language_coverage(
    coverage: Any,
    report_counts: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(coverage, dict):
        errors.append("language_coverage must be an object")
        return
    expected_keys = {"required_languages", "record_counts"}
    unknown = sorted(set(coverage) - expected_keys)
    if unknown:
        errors.append(f"language_coverage unknown keys: {', '.join(unknown)}")
    missing = sorted(expected_keys - set(coverage))
    if missing:
        errors.append(f"language_coverage missing keys: {', '.join(missing)}")
        return

    required = coverage.get("required_languages")
    if not isinstance(required, list) or not required or len(required) != len(set(required)):
        errors.append("required_languages must be a non-empty unique list")
        return
    for lang in required:
        if not isinstance(lang, str) or not lang:
            errors.append("required_languages must contain non-empty strings")
            return
        if _LANGUAGE_ID.fullmatch(lang) is None:
            errors.append(f"required_languages: {lang!r} does not match " f"^[A-Za-z0-9_-]+$")
            return

    counts = coverage.get("record_counts")
    if not isinstance(counts, dict):
        errors.append("record_counts must be an object")
        return

    for lang in required:
        if lang not in counts:
            errors.append(f"record_counts missing required language {lang!r}")
            continue
        count = counts[lang]
        if isinstance(count, bool):
            errors.append(f"record_counts.{lang} must not be a boolean")
            continue
        if not isinstance(count, int) or count < 1:
            errors.append(f"record_counts.{lang} must be a positive integer")
            continue
        report_count = report_counts.get(lang)
        if isinstance(report_count, dict):
            rc = report_count.get("record_count")
            if isinstance(rc, int) and not isinstance(rc, bool) and rc != count:
                errors.append(
                    f"record_counts.{lang}={count} does not match "
                    f"evaluation report record_count={rc}"
                )

    for lang in counts:
        if not isinstance(lang, str) or not lang:
            errors.append("record_counts keys must be non-empty strings")
            continue
        if _LANGUAGE_ID.fullmatch(lang) is None:
            errors.append(f"record_counts key {lang!r} does not match ^[A-Za-z0-9_-]+$")
            continue
        count = counts[lang]
        if isinstance(count, bool):
            errors.append(f"record_counts.{lang} must not be a boolean")
            continue
        if not isinstance(count, int) or count < 0:
            errors.append(f"record_counts.{lang} must be a non-negative integer")


def _validate_evaluation_input(
    root: Path,
    evaluation_input: Any,
    report: Any,
    errors: list[str],
) -> None:
    input_path = _file_reference(root, evaluation_input, "evaluation_input", errors)
    if input_path is None:
        return
    try:
        records = load_evaluation_records(input_path)
    except Exception as exc:
        errors.append(f"evaluation_input: {exc}")
        return

    dataset_digest = compute_evaluation_dataset_sha256(records)
    if not isinstance(report, dict):
        return
    report_ds = report.get("input_dataset_sha256")
    if dataset_digest != report_ds:
        errors.append(
            f"evaluation_input dataset digest {dataset_digest} does not match "
            f"report input_dataset_sha256 {report_ds}"
        )
        return

    per_language = report.get("per_language")
    if not isinstance(per_language, dict):
        return
    tokenizer_names = report.get("tokenizer_names", [])
    if not isinstance(tokenizer_names, list) or not tokenizer_names:
        return
    for tname in tokenizer_names:
        lang_data = per_language.get(tname)
        if not isinstance(lang_data, dict):
            continue
        for lang, metrics in lang_data.items():
            if not isinstance(metrics, dict):
                continue
            report_rc = metrics.get("record_count")
            if not isinstance(report_rc, int) or isinstance(report_rc, bool):
                continue
            actual_count = sum(1 for r in records if r.language == lang)
            if actual_count != report_rc:
                errors.append(
                    f"evaluation_input language {lang!r} has {actual_count} "
                    f"records but report per_language.{tname}.{lang}"
                    f".record_count={report_rc}"
                )


def _validate_fingerprint_linkage(
    tokenizer_manifest: dict[str, Any],
    report: Any,
    decision: Any,
    tokenizer_name: str | None,
    errors: list[str],
) -> None:
    if not isinstance(report, dict):
        return
    if not isinstance(decision, dict):
        return

    manifest_fp = tokenizer_manifest.get("fingerprint")

    report_fps = report.get("tokenizer_fingerprints")
    if not isinstance(report_fps, dict):
        return
    report_fp = report_fps.get(tokenizer_name)
    if tokenizer_name is not None and report_fp is None:
        errors.append(f"report tokenizer_fingerprints missing {tokenizer_name!r}")

    decision_fp = decision.get("tokenizer_fingerprint")

    if tokenizer_name is not None and tokenizer_name not in report.get("tokenizer_names", []):
        errors.append(f"tokenizer_name {tokenizer_name!r} not in report tokenizer_names")

    if manifest_fp is not None and report_fp is not None and manifest_fp != report_fp:
        errors.append(
            f"manifest tokenizer.fingerprint {manifest_fp} does not match "
            f"report tokenizer_fingerprints[{tokenizer_name!r}] {report_fp}"
        )
    if manifest_fp is not None and decision_fp is not None and manifest_fp != decision_fp:
        errors.append(
            f"manifest tokenizer.fingerprint {manifest_fp} does not match "
            f"decision tokenizer_fingerprint {decision_fp}"
        )
    if report_fp is not None and decision_fp is not None and report_fp != decision_fp:
        errors.append(
            f"report tokenizer_fingerprint {report_fp} does not match "
            f"decision tokenizer_fingerprint {decision_fp}"
        )


def _validate_decision(
    decision: Any,
    report: Any,
    threshold_configuration: ThresholdConfiguration | None,
    errors: list[str],
) -> str | None:
    if not isinstance(decision, dict):
        errors.append("acceptance_decision must be an object")
        return None
    tokenizer_name = decision.get("tokenizer_name")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        errors.append("acceptance_decision tokenizer_name must be a non-empty string")
        return None
    if not isinstance(report, dict) or threshold_configuration is None:
        return tokenizer_name
    try:
        expected = evaluate_tokenizer_acceptance(
            report,
            tokenizer_name,
            threshold_configuration,
        )
    except Exception as exc:
        errors.append(f"acceptance_decision cannot be recomputed: {exc}")
        return tokenizer_name
    if decision != expected:
        errors.append("acceptance_decision does not match recomputed decision")
    return tokenizer_name


def _check_accepted_evidence(
    manifest: dict[str, Any],
    bpe_tokenizer: BPETokenizer | None,
    decision: Any,
    threshold_configuration: ThresholdConfiguration | None,
    input_path: Path | None,
    loaded_tokenizer: Any,
    errors: list[str],
) -> None:
    if errors:
        return
    complete = _byte_alphabet_complete(bpe_tokenizer)
    tokenizer_manifest = manifest.get("tokenizer")
    if not isinstance(tokenizer_manifest, dict):
        tokenizer_manifest = {}
    if tokenizer_manifest.get("byte_alphabet_complete") is not True or not complete:
        errors.append("accepted evidence requires independently verified byte coverage")
    if not isinstance(decision, dict) or decision.get("passed") is not True:
        errors.append("accepted evidence requires a passing acceptance decision")
    if threshold_configuration is None:
        errors.append("accepted evidence requires a valid threshold configuration")
    else:
        if threshold_configuration.status != "production":
            errors.append("accepted evidence requires production thresholds")
        if threshold_configuration.evidence_scope != _ACCEPTED_EVIDENCE_SCOPE:
            errors.append("accepted evidence requires approved-evaluation-set evidence scope")
    if input_path is None or loaded_tokenizer is None:
        errors.append("accepted evidence is not independently verifiable")


def validate_production_evidence(manifest_path: Path) -> ProductionEvidenceValidation:
    errors: list[str] = []
    manifest = _load_canonical_json(manifest_path, errors, "manifest")
    manifest_digest = _sha256(manifest_path) if manifest_path.is_file() else "0" * 64
    status = manifest.get("status", "invalid") if isinstance(manifest, dict) else "invalid"
    if not _validate_manifest_shape(manifest, errors):
        return ProductionEvidenceValidation(
            manifest_digest,
            str(status),
            False,
            False,
            tuple(errors),
        )

    assert isinstance(manifest, dict)
    root = manifest_path.parent.resolve()

    tokenizer_manifest = _validate_tokenizer_section(manifest.get("tokenizer"), errors)
    loaded_tokenizer, bpe_tokenizer = _validate_tokenizer_artifact(
        root,
        tokenizer_manifest,
        errors,
    )

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

    decision: Any | None = None
    if decision_path is not None:
        decision = _load_canonical_json(decision_path, errors, "acceptance_decision")

    threshold_configuration: ThresholdConfiguration | None = None
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

    tokenizer_name = _validate_decision(
        decision,
        report,
        threshold_configuration,
        errors,
    )

    _validate_fingerprint_linkage(
        tokenizer_manifest,
        report,
        decision,
        tokenizer_name,
        errors,
    )

    _validate_evaluation_input(root, manifest.get("evaluation_input"), report, errors)

    report_counts: dict[str, Any] = {}
    if isinstance(report, dict) and tokenizer_name is not None:
        per_language = report.get("per_language")
        if isinstance(per_language, dict):
            tokenizer_languages = per_language.get(tokenizer_name)
            if isinstance(tokenizer_languages, dict):
                report_counts = tokenizer_languages

    _validate_language_coverage(
        manifest.get("language_coverage"),
        report_counts,
        errors,
    )

    if status == "accepted":
        _check_accepted_evidence(
            manifest,
            bpe_tokenizer,
            decision,
            threshold_configuration,
            input_path,
            loaded_tokenizer,
            errors,
        )

    is_valid = not errors
    is_accepted = status == "accepted" and is_valid

    return ProductionEvidenceValidation(
        manifest_digest,
        str(status),
        is_valid,
        is_accepted,
        tuple(errors),
    )
