from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_receipt_manifest as module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    receipt_directory = tmp_path / "receipt-evidence"
    receipt_directory.mkdir()
    acceptance_directory = receipt_directory / "accepted-promotion"
    acceptance_directory.mkdir()
    receipt_path = receipt_directory / "acceptance-receipt.json"
    receipt_path.write_text('{"accepted": true}', encoding="utf-8")
    manifest_path = tmp_path / "acceptance-receipt-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "operator": "operator@example.com",
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "reviewer": "reviewer@example.com",
                "schema_version": "tokenizer-promotion-acceptance-receipt-manifest-v1",
            }
        ),
        encoding="utf-8",
    )
    return receipt_directory, receipt_path, manifest_path


def test_verifies_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    receipt = SimpleNamespace(
        operator="operator@example.com",
        reviewer="reviewer@example.com",
    )
    verified = SimpleNamespace(receipt=receipt)
    calls: list[Path] = []

    def verifier(directory: Path) -> object:
        calls.append(directory)
        return verified

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_directory",
        verifier,
    )

    result = module.verify_promotion_acceptance_receipt_manifest(
        receipt_directory,
        manifest_path,
    )

    assert result.receipt_directory is verified
    assert result.operator == "operator@example.com"
    assert result.reviewer == "reviewer@example.com"
    assert calls == [receipt_directory]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "unsupported", "unsupported"),
        ("receipt_sha256", "0" * 64, "receipt_sha256"),
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
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = SimpleNamespace(
        receipt=SimpleNamespace(
            operator="operator@example.com",
            reviewer="reviewer@example.com",
        )
    )
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_directory",
        lambda _: verified,
    )

    with pytest.raises(ValueError, match=message):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )


def test_rejects_missing_or_unexpected_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_directory",
        lambda _: SimpleNamespace(receipt=SimpleNamespace()),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("reviewer")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )

    manifest["reviewer"] = "reviewer@example.com"
    manifest["extra"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )


@pytest.mark.parametrize("content", ["not-json", "[]"])
def test_rejects_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    manifest_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_directory",
        lambda _: SimpleNamespace(receipt=SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="JSON"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )


def test_rejects_missing_or_symlinked_manifest(tmp_path: Path) -> None:
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )

    target = tmp_path / "manifest-target.json"
    target.write_text("{}", encoding="utf-8")
    manifest_path.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )


def test_rejects_directory_in_place_of_manifest(tmp_path: Path) -> None:
    receipt_directory, _, manifest_path = _fixture(tmp_path)
    manifest_path.unlink()
    manifest_path.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        module.verify_promotion_acceptance_receipt_manifest(
            receipt_directory,
            manifest_path,
        )
