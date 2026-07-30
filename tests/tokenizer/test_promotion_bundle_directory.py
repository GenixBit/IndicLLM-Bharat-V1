from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_bundle_directory as module


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "package").mkdir()
    (bundle / "receipt.json").write_text("{}", encoding="utf-8")
    return bundle


def test_verifies_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    receipt = SimpleNamespace(operator="reviewer@example.com")
    monkeypatch.setattr(module, "verify_promotion_receipt", lambda *_: receipt)

    result = module.verify_promotion_bundle_directory(bundle)

    assert result.bundle_directory == bundle
    assert result.receipt is receipt


def test_passes_expected_paths_to_receipt_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def verify(package_directory: Path, receipt_path: Path) -> object:
        calls.append((package_directory, receipt_path))
        return object()

    monkeypatch.setattr(module, "verify_promotion_receipt", verify)

    module.verify_promotion_bundle_directory(bundle)

    assert calls == [(bundle / "package", bundle / "receipt.json")]


@pytest.mark.parametrize("missing", ["package", "receipt.json"])
def test_rejects_missing_required_entry(tmp_path: Path, missing: str) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / missing
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()

    with pytest.raises(ValueError, match="missing required entries"):
        module.verify_promotion_bundle_directory(bundle)


def test_rejects_unexpected_entry(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected entries"):
        module.verify_promotion_bundle_directory(bundle)


def test_rejects_symlinked_bundle(tmp_path: Path) -> None:
    target = _bundle(tmp_path)
    link = tmp_path / "bundle-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must be a regular directory"):
        module.verify_promotion_bundle_directory(link)


def test_rejects_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a regular directory"):
        module.verify_promotion_bundle_directory(tmp_path / "missing")


def test_rejects_symlinked_package(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    package = bundle / "package"
    package.rmdir()
    target = tmp_path / "package-target"
    target.mkdir()
    package.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="package must be a regular directory"):
        module.verify_promotion_bundle_directory(bundle)


def test_rejects_non_directory_package(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    package = bundle / "package"
    package.rmdir()
    package.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="package must be a regular directory"):
        module.verify_promotion_bundle_directory(bundle)


def test_rejects_symlinked_receipt(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    receipt = bundle / "receipt.json"
    receipt.unlink()
    target = tmp_path / "receipt-target.json"
    target.write_text("{}", encoding="utf-8")
    receipt.symlink_to(target)

    with pytest.raises(ValueError, match="receipt must be a regular file"):
        module.verify_promotion_bundle_directory(bundle)


def test_rejects_non_file_receipt(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    receipt = bundle / "receipt.json"
    receipt.unlink()
    receipt.mkdir()

    with pytest.raises(ValueError, match="receipt must be a regular file"):
        module.verify_promotion_bundle_directory(bundle)
