from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance as module


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "package").mkdir()
    receipt = bundle / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "accepted": True,
                "operator": "operator@example.com",
                "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "reviewer": "reviewer@example.com",
                "schema_version": "tokenizer-promotion-acceptance-v1",
            }
        ),
        encoding="utf-8",
    )
    return bundle, acceptance


def _stub_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    verified = SimpleNamespace(receipt=SimpleNamespace(operator="operator@example.com"))
    monkeypatch.setattr(module, "verify_promotion_bundle_directory", lambda _: verified)


def test_verifies_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, acceptance = _fixture(tmp_path)
    _stub_bundle(monkeypatch)

    result = module.verify_promotion_acceptance(bundle, acceptance)

    assert result.operator == "operator@example.com"
    assert result.reviewer == "reviewer@example.com"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accepted", False, "explicitly accept"),
        ("receipt_sha256", "0" * 64, "receipt_sha256 does not match"),
        ("operator", "other@example.com", "operator does not match"),
        ("reviewer", "operator@example.com", "reviewer must differ"),
        ("reviewer", "", "reviewer must be non-empty"),
        ("schema_version", "v2", "unsupported"),
    ],
)
def test_rejects_invalid_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle, acceptance = _fixture(tmp_path)
    _stub_bundle(monkeypatch)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    payload[field] = value
    acceptance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.verify_promotion_acceptance(bundle, acceptance)


def test_rejects_missing_or_extra_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, acceptance = _fixture(tmp_path)
    _stub_bundle(monkeypatch)
    payload = json.loads(acceptance.read_text(encoding="utf-8"))
    payload["extra"] = True
    acceptance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        module.verify_promotion_acceptance(bundle, acceptance)


def test_rejects_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, acceptance = _fixture(tmp_path)
    _stub_bundle(monkeypatch)
    acceptance.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        module.verify_promotion_acceptance(bundle, acceptance)


def test_rejects_missing_or_symlinked_acceptance(tmp_path: Path) -> None:
    bundle, acceptance = _fixture(tmp_path)
    acceptance.unlink()
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance(bundle, acceptance)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    acceptance.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance(bundle, acceptance)
