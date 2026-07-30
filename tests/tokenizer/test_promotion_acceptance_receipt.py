from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_receipt as module


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "accepted-promotion"
    directory.mkdir()
    (directory / "bundle").mkdir()
    acceptance = directory / "acceptance.json"
    acceptance.write_text("accepted", encoding="utf-8")
    receipt = tmp_path / "acceptance-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "acceptance_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
                "operator": "operator@example.com",
                "reviewer": "reviewer@example.com",
                "schema_version": "tokenizer-promotion-acceptance-receipt-v1",
            }
        ),
        encoding="utf-8",
    )
    return directory, receipt


def _stub_verifier(monkeypatch: pytest.MonkeyPatch) -> object:
    acceptance = SimpleNamespace(
        operator="operator@example.com",
        reviewer="reviewer@example.com",
    )
    verified = SimpleNamespace(acceptance=acceptance)
    monkeypatch.setattr(module, "verify_promotion_acceptance_directory", lambda _: verified)
    return verified


def test_verifies_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, receipt = _fixture(tmp_path)
    verified = _stub_verifier(monkeypatch)

    result = module.verify_promotion_acceptance_receipt(directory, receipt)

    assert result.acceptance_directory is verified
    assert result.operator == "operator@example.com"
    assert result.reviewer == "reviewer@example.com"


def test_delegates_exact_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, receipt = _fixture(tmp_path)
    calls: list[Path] = []

    def verifier(path: Path) -> object:
        calls.append(path)
        return SimpleNamespace(
            acceptance=SimpleNamespace(
                operator="operator@example.com",
                reviewer="reviewer@example.com",
            )
        )

    monkeypatch.setattr(module, "verify_promotion_acceptance_directory", verifier)
    module.verify_promotion_acceptance_receipt(directory, receipt)
    assert calls == [directory]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "v2", "unsupported"),
        ("acceptance_sha256", "0" * 64, "acceptance_sha256"),
        ("operator", "other@example.com", "operator does not match"),
        ("reviewer", "other@example.com", "reviewer does not match"),
    ],
)
def test_rejects_mismatched_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    directory, receipt = _fixture(tmp_path)
    _stub_verifier(monkeypatch)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = value
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.verify_promotion_acceptance_receipt(directory, receipt)


def test_rejects_unexpected_or_missing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, receipt = _fixture(tmp_path)
    _stub_verifier(monkeypatch)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["extra"] = True
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        module.verify_promotion_acceptance_receipt(directory, receipt)


def test_rejects_invalid_json_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, receipt = _fixture(tmp_path)
    _stub_verifier(monkeypatch)

    receipt.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        module.verify_promotion_acceptance_receipt(directory, receipt)

    receipt.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module.verify_promotion_acceptance_receipt(directory, receipt)


def test_rejects_missing_or_symlinked_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, receipt = _fixture(tmp_path)
    _stub_verifier(monkeypatch)
    receipt.unlink()
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance_receipt(directory, receipt)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    receipt.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance_receipt(directory, receipt)
