from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.tokenizer.bpe import BPETokenizer
from bharat.tokenizer.bpe_adapter import BharatBPETokenizer
from bharat.tokenizer.evaluation import (
    TokenizerEvaluation,
)
from scripts.build_production_tokenizer_evidence import main


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _build_bpe_tokenizer(tmp_path: Path) -> Path:
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
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    return path


def _build_input_jsonl(tmp_path: Path) -> Path:
    records = [
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
    ]
    lines = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    path = tmp_path / "input.jsonl"
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def _build_thresholds(tmp_path: Path) -> Path:
    payload = {
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
            "required_languages": ["en"],
            "min_records_per_required_language": 1,
        },
    }
    path = tmp_path / "thresholds.json"
    path.write_bytes(_canonical(payload))
    return path


def _compute_real_report(
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
    path.write_bytes(_canonical(report))
    return path


def _build_decision(
    tmp_path: Path, report_path: Path, thresholds_path: Path, tokenizer_name: str, tokenizer_fp: str
) -> Path:
    from bharat.tokenizer.acceptance import ThresholdConfiguration, evaluate_tokenizer_acceptance

    report = json.loads(report_path.read_text(encoding="utf-8"))
    thresh = json.loads(thresholds_path.read_text(encoding="utf-8"))
    config = ThresholdConfiguration.from_payload(thresh)
    decision = evaluate_tokenizer_acceptance(report, tokenizer_name, config)
    path = tmp_path / "decision.json"
    path.write_bytes(_canonical(decision))
    return path


@pytest.fixture
def cli_fixtures(tmp_path: Path) -> dict[str, Path]:
    tokenizer_path = _build_bpe_tokenizer(tmp_path)
    input_path = _build_input_jsonl(tmp_path)
    thresholds_path = _build_thresholds(tmp_path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_fp = tokenizer.compute_hash()
    report_path = _compute_real_report(tmp_path, tokenizer_path, input_path, "test-bpe")
    decision_path = _build_decision(
        tmp_path, report_path, thresholds_path, "test-bpe", tokenizer_fp
    )
    return {
        "tokenizer_path": tokenizer_path,
        "input_path": input_path,
        "thresholds_path": thresholds_path,
        "report_path": report_path,
        "decision_path": decision_path,
    }


def _build_args(
    tmp_path: Path,
    fixtures: dict[str, Path],
    output_name: str = "manifest.json",
) -> list[str]:
    return [
        "--evidence-root",
        str(tmp_path),
        "--repository-commit-sha",
        "a" * 40,
        "--tokenizer",
        str(fixtures["tokenizer_path"]),
        "--evaluation-input",
        str(fixtures["input_path"]),
        "--evaluation-report",
        str(fixtures["report_path"]),
        "--acceptance-decision",
        str(fixtures["decision_path"]),
        "--threshold-configuration",
        str(fixtures["thresholds_path"]),
        "--generating-command",
        "test-command",
        "--output",
        str(tmp_path / output_name),
    ]


# -- CLI success --


def test_cli_success(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    rc = main(_build_args(tmp_path, cli_fixtures))
    assert rc == 0
    out = tmp_path / "manifest.json"
    assert out.exists()


# -- CLI validation failure --


def test_cli_validation_failure(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    args = _build_args(tmp_path, cli_fixtures)
    bad_report = tmp_path / "bad_report.json"
    bad_report.write_text("not json", encoding="utf-8")
    idx = args.index(str(cli_fixtures["report_path"]))
    args[idx] = str(bad_report)
    rc = main(args)
    assert rc == 2


# -- CLI existing-output failure --


def test_cli_existing_output_failure(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    out_file = tmp_path / "manifest.json"
    out_file.write_text("existing", encoding="utf-8")
    args = _build_args(tmp_path, cli_fixtures)
    rc = main(args)
    assert rc == 3
    assert out_file.read_text(encoding="utf-8") == "existing"


# -- CLI outputs manifest_sha256 --


def test_cli_outputs_digest(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    cap = _capture_main(_build_args(tmp_path, cli_fixtures))
    assert "manifest_sha256" in cap


def _capture_main(argv: list[str]) -> str:
    import io
    import sys as _sys

    old_out = _sys.stdout
    try:
        _sys.stdout = io.StringIO()
        rc = main(argv)
        assert rc == 0
        return _sys.stdout.getvalue()
    finally:
        _sys.stdout = old_out


# -- Offline execution --


def test_cli_offline(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    rc = main(_build_args(tmp_path, cli_fixtures))
    assert rc == 0


# -- Determinism across CLI invocations --


def test_cli_deterministic(tmp_path: Path, cli_fixtures: dict[str, Path]) -> None:
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    args1 = _build_args(tmp_path, cli_fixtures, output_name="out1.json")
    args2 = _build_args(tmp_path, cli_fixtures, output_name="out2.json")
    assert main(args1) == 0
    assert main(args2) == 0
    assert out1.read_bytes() == out2.read_bytes()
