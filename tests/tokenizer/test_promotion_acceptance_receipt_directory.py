from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_receipt_directory as module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    directory = tmp_path / "evidence"
    directory.mkdir()
    acceptance_directory = directory / "accepted-promotion"
    acceptance_directory.mkdir()
    receipt = directory / "acceptance-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    return directory, acceptance_directory, receipt


def test_verifies_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory, acceptance_directory, receipt = _fixture(tmp_path)
    verified = SimpleNamespace(operator="operator@example.com")
    calls: list[tuple[Path, Path]] = []

    def verifier(directory_path: Path, receipt_path: Path) -> object:
        calls.append((directory_path, receipt_path))
        return verified

    monkeypatch.setattr(module, "verify_promotion_acceptance_receipt", verifier)

    result = module.verify_promotion_acceptance_receipt_directory(directory)

    assert result.directory == directory
    assert result.receipt is verified
    assert calls == [(acceptance_directory, receipt)]


@pytest.mark.parametrize("missing_name", ["accepted-promotion", "acceptance-receipt.json"])
def test_rejects_missing_entries(tmp_path: Path, missing_name: str) -> None:
    directory, acceptance_directory, receipt = _fixture(tmp_path)
    path = directory / missing_name
    if path == acceptance_directory:
        path.rmdir()
    else:
        receipt.unlink()

    with pytest.raises(ValueError, match="missing entries"):
        module.verify_promotion_acceptance_receipt_directory(directory)


def test_rejects_unexpected_entries(tmp_path: Path) -> None:
    directory, _, _ = _fixture(tmp_path)
    (directory / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected entries"):
        module.verify_promotion_acceptance_receipt_directory(directory)


def test_rejects_missing_or_symlinked_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_receipt_directory(missing)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_receipt_directory(link)


def test_rejects_invalid_acceptance_directory(tmp_path: Path) -> None:
    directory, acceptance_directory, _ = _fixture(tmp_path)
    acceptance_directory.rmdir()
    acceptance_directory.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="accepted promotion evidence"):
        module.verify_promotion_acceptance_receipt_directory(directory)


def test_rejects_symlinked_acceptance_directory(tmp_path: Path) -> None:
    directory, acceptance_directory, _ = _fixture(tmp_path)
    acceptance_directory.rmdir()
    target = tmp_path / "accepted-target"
    target.mkdir()
    acceptance_directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="accepted promotion evidence"):
        module.verify_promotion_acceptance_receipt_directory(directory)


def test_rejects_invalid_receipt(tmp_path: Path) -> None:
    directory, _, receipt = _fixture(tmp_path)
    receipt.unlink()
    receipt.mkdir()

    with pytest.raises(ValueError, match="receipt must be a regular file"):
        module.verify_promotion_acceptance_receipt_directory(directory)


def test_rejects_symlinked_receipt(tmp_path: Path) -> None:
    directory, _, receipt = _fixture(tmp_path)
    receipt.unlink()
    target = tmp_path / "receipt-target.json"
    target.write_text("{}", encoding="utf-8")
    receipt.symlink_to(target)

    with pytest.raises(ValueError, match="receipt must be a regular file"):
        module.verify_promotion_acceptance_receipt_directory(directory)
