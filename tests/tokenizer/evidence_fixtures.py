from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_bpe_tokenizer(tmp_path: Path, name: str = "tokenizer.json") -> Path:
    byte_value_to_id = {b: 4 + b for b in range(256)}
    id_to_bytes = {4 + b: bytes([b]) for b in range(256)}
    special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab = dict(special_tokens)
    for b in range(256):
        vocab[f"<byte_{b:02x}>"] = 4 + b
    tok = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens=special_tokens,
        reserved_tokens={},
        byte_value_to_id=byte_value_to_id,
        id_to_bytes=id_to_bytes,
        vocab=vocab,
        merges=(),
        tokenizer_hash="",
    )
    tok.tokenizer_hash = tok.compute_hash()
    tok.validate()
    path = tmp_path / name
    tok.save(path)
    return path


def build_bad_bpe_tokenizer(
    tmp_path: Path,
    name: str,
    missing_byte: int | None = None,
    duplicate_byte: bool = False,
    bad_byte_mapping: bool = False,
    collision_special: bool = False,
) -> tuple[Path, str]:
    byte_value_to_id = {b: 4 + b for b in range(256)}
    id_to_bytes = {4 + b: bytes([b]) for b in range(256)}
    special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    if missing_byte is not None:
        del byte_value_to_id[missing_byte]
    if duplicate_byte:
        byte_value_to_id[0] = 4 + 1
    if bad_byte_mapping:
        id_to_bytes[4 + 0] = bytes([1])
    if collision_special:
        byte_value_to_id[0] = 1
        id_to_bytes[1] = bytes([0])
    vocab = dict(special_tokens)
    for b in range(256):
        if b in byte_value_to_id:
            vocab[f"<byte_{b:02x}>"] = byte_value_to_id[b]
    tok = BPETokenizer(
        schema_version="bpe-v1",
        normalization="nfc",
        special_tokens=special_tokens,
        reserved_tokens={},
        byte_value_to_id=byte_value_to_id,
        id_to_bytes=id_to_bytes,
        vocab=vocab,
        merges=(),
        tokenizer_hash="",
    )
    tok.tokenizer_hash = tok.compute_hash()
    data = tok.to_dict()
    path = tmp_path / name
    path.write_bytes(canonical_bytes(data))
    return path, tok.compute_hash()


def build_input_jsonl(
    tmp_path: Path,
    overrides: list[dict[str, Any]] | None = None,
    name: str = "input.jsonl",
) -> Path:
    records = overrides or [
        {
            "id": "rec-1",
            "language": "en",
            "script": "Latin",
            "domain": "web",
            "text": "Hello world",
        },
        {
            "id": "rec-2",
            "language": "hi",
            "script": "Devanagari",
            "domain": "web",
            "text": "\u0928\u092e\u0938\u094d\u0924\u0947 \u092d\u093e\u0930\u0924",
        },
        {
            "id": "rec-3",
            "language": "en",
            "script": "Latin",
            "domain": "news",
            "text": "Good morning",
        },
        {
            "id": "rec-4",
            "language": "hi",
            "script": "Devanagari",
            "domain": "canonical",
            "text": "\u092d\u093e\u0930\u0924",
            "canonical_equivalent": "\u092d\u093e\u0930\u0924",
        },
    ]
    lines = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    path = tmp_path / name
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def build_production_thresholds(tmp_path: Path, name: str = "thresholds.json") -> Path:
    payload = {
        "schema_version": "tokenizer-acceptance-thresholds-v1",
        "status": "production",
        "evidence_scope": "approved-evaluation-set",
        "notes": [],
        "thresholds": {
            "min_record_count": 1,
            "min_required_round_trip_rate": 1.0,
            "min_canonical_pass_rate": 1.0,
            "min_canonical_evaluated_count": 1,
            "max_unknown_token_rate": 0.0,
            "require_complete_byte_coverage": True,
            "required_languages": ["en", "hi"],
            "min_records_per_required_language": 1,
        },
    }
    path = tmp_path / name
    path.write_bytes(canonical_bytes(payload))
    return path


def compute_real_report(
    tmp_path: Path,
    tokenizer_path: Path,
    input_path: Path,
    tokenizer_name: str = "test-bpe",
    name: str = "report.json",
) -> Path:
    loaded = BPETokenizer.load(tokenizer_path)
    adapter = BharatBPETokenizer(loaded)
    evaluation = TokenizerEvaluation({tokenizer_name: adapter})
    evaluation.load_records(input_path)
    report = evaluation.compute()
    path = tmp_path / name
    path.write_bytes(canonical_bytes(report))
    return path


def build_acceptance_decision(
    tmp_path: Path,
    report_path: Path,
    thresholds_path: Path,
    tokenizer_name: str,
    name: str = "decision.json",
) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresh = json.loads(thresholds_path.read_text(encoding="utf-8"))
    config = ThresholdConfiguration.from_payload(thresh)
    decision = evaluate_tokenizer_acceptance(report, tokenizer_name, config)
    path = tmp_path / name
    path.write_bytes(canonical_bytes(decision))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence_fixtures(tmp_path: Path) -> dict[str, Any]:
    tokenizer_path = build_bpe_tokenizer(tmp_path)
    input_path = build_input_jsonl(tmp_path)
    thresholds_path = build_production_thresholds(tmp_path)
    report_path = compute_real_report(tmp_path, tokenizer_path, input_path, "test-bpe")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    decision_path = build_acceptance_decision(tmp_path, report_path, thresholds_path, "test-bpe")
    return {
        "tokenizer_path": tokenizer_path,
        "input_path": input_path,
        "thresholds_path": thresholds_path,
        "report_path": report_path,
        "decision_path": decision_path,
        "tokenizer_fp": tokenizer_fp,
        "tokenizer_name": "test-bpe",
    }
