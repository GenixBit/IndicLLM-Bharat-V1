from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.loader import load_tokenizer
from bharat.tokenizer.production_evidence import validate_production_evidence

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
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if raw != _canonical_bytes(value):
        raise ValueError(f"{label} JSON bytes are not canonical")
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
    if path.is_symlink() and resolved.parent != path.parent.resolve():
        raise ValueError(f"{label} symlink must remain inside evidence root")
    return resolved, relative.as_posix()


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
        raise ValueError(
            "repository_commit_sha must be a lowercase 40- or 64-character hex ID"
        )
    if not generating_commands or any(
        not isinstance(item, str) or not item.strip() for item in generating_commands
    ):
        raise ValueError("generating_commands must contain non-empty strings")

    root = evidence_root.resolve()
    if not root.is_dir():
        raise ValueError("evidence_root must be an existing directory")

    tokenizer_file, tokenizer_relative = _relative_file(
        root, tokenizer_path, "tokenizer_path"
    )
    input_file, input_relative = _relative_file(
        root, evaluation_input_path, "evaluation_input_path"
    )
    report_file, report_relative = _relative_file(
        root, evaluation_report_path, "evaluation_report_path"
    )
    decision_file, decision_relative = _relative_file(
        root,
        acceptance_decision_path,
        "acceptance_decision_path",
    )
    thresholds_file, thresholds_relative = _relative_file(
        root,
        threshold_configuration_path,
        "threshold_configuration_path",
    )

    report = _load_canonical_json(report_file, "evaluation_report")
    decision = _load_canonical_json(decision_file, "acceptance_decision")
    _load_canonical_json(thresholds_file, "threshold_configuration")

    tokenizer_name = decision.get("tokenizer_name")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError(
            "acceptance_decision tokenizer_name must be a non-empty string"
        )

    per_language = report.get("per_language")
    if not isinstance(per_language, dict):
        raise ValueError("evaluation_report per_language must be an object")
    tokenizer_languages = per_language.get(tokenizer_name)
    if not isinstance(tokenizer_languages, dict) or not tokenizer_languages:
        raise ValueError(
            "evaluation_report must contain per-language metrics for tokenizer_name"
        )

    record_counts: dict[str, int] = {}
    for language, metrics in tokenizer_languages.items():
        if not isinstance(language, str) or not isinstance(metrics, dict):
            raise ValueError("evaluation_report per-language entries are invalid")
        count = metrics.get("record_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"evaluation_report record_count for {language!r} must be positive"
            )
        record_counts[language] = count

    loaded = load_tokenizer(tokenizer_file)
    bpe = BPETokenizer.load(tokenizer_file)
    byte_alphabet_complete = set(bpe.byte_value_to_id) == set(range(256)) and len(
        set(bpe.byte_value_to_id.values())
    ) == 256
    if not byte_alphabet_complete:
        raise ValueError("tokenizer artifact does not contain a complete byte alphabet")

    return {
        "schema_version": "tokenizer-production-evidence-manifest-v1",
        "status": "candidate",
        "evidence_scope": "production-local-approved",
        "repository_commit_sha": repository_commit_sha,
        "tokenizer": {
            "artifact_path": tokenizer_relative,
            "artifact_sha256": _sha256(tokenizer_file),
            "fingerprint": loaded.fingerprint(),
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
            "record_counts": {
                key: record_counts[key] for key in sorted(record_counts)
            },
        },
        "generating_commands": list(generating_commands),
    }


def write_candidate_manifest(output_path: Path, **kwargs: Any) -> str:
    """Validate and publish a canonical candidate manifest without overwriting files."""

    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    manifest = build_candidate_manifest(**kwargs)
    payload = _canonical_bytes(manifest)

    temp = output.with_name(f".{output.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        validation = validate_production_evidence(temp)
        if not validation.valid:
            raise ValueError(
                "candidate evidence validation failed: "
                + "; ".join(validation.errors)
            )
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        temp.unlink(missing_ok=True)

    return hashlib.sha256(payload).hexdigest()
