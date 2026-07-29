from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bharat.tokenizer.acceptance import (
    ThresholdConfiguration,
    evaluate_tokenizer_acceptance,
)
from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.bpe_adapter import BharatBPETokenizer
from bharat.tokenizer.evaluation import (
    TokenizerEvaluation,
)
from bharat.tokenizer.production_evidence import validate_production_evidence
from bharat.tokenizer.production_evidence_builder import write_candidate_manifest


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _build_tokenizer(root: Path) -> Path:
    byte_value_to_id = {value: 4 + value for value in range(256)}
    tokenizer = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens={"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
        reserved_tokens={},
        byte_value_to_id=byte_value_to_id,
        id_to_bytes={4 + value: bytes([value]) for value in range(256)},
        vocab={
            **{"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
            **{f"<byte_{value:02x}>": 4 + value for value in range(256)},
        },
        merges=(),
        tokenizer_hash="",
    )
    tokenizer.tokenizer_hash = tokenizer.compute_hash()
    tokenizer.validate()
    path = root / "tokenizer.json"
    tokenizer.save(path)
    return path


def _build_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    tokenizer_path = _build_tokenizer(root)
    input_path = root / "input.jsonl"
    records = [
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
    ]
    input_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    thresholds = {
        "schema_version": "tokenizer-acceptance-thresholds-v1",
        "status": "production",
        "evidence_scope": "approved-evaluation-set",
        "notes": [],
        "thresholds": {
            "min_record_count": 1,
            "min_required_round_trip_rate": 0.0,
            "min_canonical_pass_rate": 0.0,
            "max_unknown_token_rate": 1.0,
            "require_complete_byte_coverage": False,
            "required_languages": ["en", "hi"],
            "min_records_per_required_language": 1,
        },
    }
    thresholds_path = root / "thresholds.json"
    thresholds_path.write_bytes(_canonical(thresholds))

    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_name = "test-bpe"
    adapter = BharatBPETokenizer(tokenizer)
    evaluation = TokenizerEvaluation({tokenizer_name: adapter})
    evaluation.load_records(input_path)
    report = evaluation.compute()
    report_path = root / "report.json"
    report_path.write_bytes(_canonical(report))

    configuration = ThresholdConfiguration.from_payload(thresholds)
    decision = evaluate_tokenizer_acceptance(report, tokenizer_name, configuration)
    decision_path = root / "decision.json"
    decision_path.write_bytes(_canonical(decision))
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
