from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.production_evidence import validate_production_evidence
from bharat.tokenizer.production_evidence_builder import write_candidate_manifest
from tests.tokenizer.evidence_fixtures import (
    build_acceptance_decision,
    build_bpe_tokenizer,
    build_input_jsonl,
    build_production_thresholds,
    canonical_bytes,
    compute_real_report,
)


def _canonical(value: object) -> bytes:
    return canonical_bytes(value)


def _build_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    tokenizer_path = build_bpe_tokenizer(root)
    input_path = build_input_jsonl(
        root,
        overrides=[
            {
                "id": "en-1",
                "language": "en",
                "script": "Latin",
                "domain": "web",
                "text": "Hello Bharat",
            },
            {
                "id": "hi-1",
                "language": "hi",
                "script": "Devanagari",
                "domain": "web",
                "text": "नमस्ते भारत",
            },
        ],
        name="input.jsonl",
    )
    thresholds_path = build_production_thresholds(root)
    report_path = compute_real_report(root, tokenizer_path, input_path, "test-bpe")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    decision_path = build_acceptance_decision(
        root, report_path, thresholds_path, "test-bpe", tokenizer_fp
    )
    return tokenizer_path, input_path, report_path, decision_path, thresholds_path


def _write(root: Path, output: Path) -> str:
    tokenizer, input_path, report, decision, thresholds = _build_inputs(root)
    return write_candidate_manifest(
        output,
        evidence_root=root,
        repository_commit_sha="a" * 40,
        tokenizer_path=tokenizer,
        evaluation_input_path=input_path,
        evaluation_report_path=report,
        acceptance_decision_path=decision,
        threshold_configuration_path=thresholds,
        generating_commands=["python local-evaluation.py"],
    )


def test_builder_writes_canonical_valid_candidate(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    digest = _write(tmp_path, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output.read_bytes() == _canonical(payload)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert payload["status"] == "candidate"
    assert payload["language_coverage"] == {
        "record_counts": {"en": 1, "hi": 1},
        "required_languages": ["en", "hi"],
    }
    result = validate_production_evidence(output)
    assert result.valid
    assert not result.accepted


def test_builder_is_repeatable_for_identical_inputs(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    first_digest = _write(first_root, first_root / "manifest.json")
    second_digest = _write(second_root, second_root / "manifest.json")

    assert first_digest == second_digest
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json"
    ).read_bytes()


def test_builder_rejects_file_outside_evidence_root(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    tokenizer, input_path, report, decision, thresholds = _build_inputs(root)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(input_path.read_bytes())

    with pytest.raises(ValueError, match="inside evidence root"):
        write_candidate_manifest(
            root / "manifest.json",
            evidence_root=root,
            repository_commit_sha="a" * 40,
            tokenizer_path=tokenizer,
            evaluation_input_path=outside,
            evaluation_report_path=report,
            acceptance_decision_path=decision,
            threshold_configuration_path=thresholds,
            generating_commands=["local command"],
        )

    assert not (root / "manifest.json").exists()


def test_builder_refuses_output_collision(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write(tmp_path, output)

    assert output.read_text(encoding="utf-8") == "existing"
