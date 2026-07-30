from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_receipt as module


def _receipt(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "tokenizer-promotion-receipt-v1",
        "manifest_sha256": "manifest",
        "readiness_sha256": "readiness",
        "decision_sha256": "decision",
        "operator": "reviewer@example.com",
    }
    value.update(overrides)
    return value


def _write_receipt(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _package_verification() -> SimpleNamespace:
    verified = SimpleNamespace(
        manifest_sha256="manifest",
        readiness_sha256="readiness",
        decision_sha256="decision",
        operator="reviewer@example.com",
    )
    return SimpleNamespace(package=verified)


def test_verifies_receipt_bound_to_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_directory = tmp_path / "package"
    receipt_path = _write_receipt(tmp_path, _receipt())
    package = _package_verification()
    monkeypatch.setattr(module, "verify_promotion_package_directory", lambda _: package)

    result = module.verify_promotion_receipt(package_directory, receipt_path)

    assert result.package is package
    assert result.operator == "reviewer@example.com"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("manifest_sha256", "manifest_sha256 does not match"),
        ("readiness_sha256", "readiness_sha256 does not match"),
        ("decision_sha256", "decision_sha256 does not match"),
    ],
)
def test_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    receipt_path = _write_receipt(tmp_path, _receipt(**{field: "wrong"}))
    monkeypatch.setattr(
        module,
        "verify_promotion_package_directory",
        lambda _: _package_verification(),
    )

    with pytest.raises(ValueError, match=message):
        module.verify_promotion_receipt(tmp_path / "package", receipt_path)


def test_rejects_operator_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = _write_receipt(tmp_path, _receipt(operator="someone-else"))
    monkeypatch.setattr(
        module,
        "verify_promotion_package_directory",
        lambda _: _package_verification(),
    )

    with pytest.raises(ValueError, match="operator does not match decision"):
        module.verify_promotion_receipt(tmp_path / "package", receipt_path)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "must be a JSON object"),
        ({"schema_version": "tokenizer-promotion-receipt-v1"}, "unexpected or missing"),
        (_receipt(schema_version="other"), "unsupported promotion receipt schema"),
        (_receipt(operator=""), "operator must be non-empty"),
    ],
)
def test_rejects_invalid_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    message: str,
) -> None:
    receipt_path = _write_receipt(tmp_path, value)
    monkeypatch.setattr(
        module,
        "verify_promotion_package_directory",
        lambda _: _package_verification(),
    )

    with pytest.raises(ValueError, match=message):
        module.verify_promotion_receipt(tmp_path / "package", receipt_path)


def test_rejects_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "verify_promotion_package_directory",
        lambda _: _package_verification(),
    )

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        module.verify_promotion_receipt(tmp_path / "package", receipt_path)


def test_rejects_symlinked_receipt(tmp_path: Path) -> None:
    target = _write_receipt(tmp_path, _receipt())
    link = tmp_path / "receipt-link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must be a regular file"):
        module.verify_promotion_receipt(tmp_path / "package", link)


def test_rejects_missing_receipt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a regular file"):
        module.verify_promotion_receipt(
            tmp_path / "package",
            tmp_path / "missing.json",
        )
