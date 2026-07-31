from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_receipt_manifest_directory as module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "manifest-evidence"
    root.mkdir()
    receipt_directory = root / "receipt-evidence"
    receipt_directory.mkdir()
    manifest_path = root / "acceptance-receipt-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return root, receipt_directory, manifest_path


def test_verifies_exact_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, receipt_directory, manifest_path = _fixture(tmp_path)
    verified = SimpleNamespace(operator="operator@example.com")
    calls: list[tuple[Path, Path]] = []

    def verifier(receipt: Path, manifest: Path) -> object:
        calls.append((receipt, manifest))
        return verified

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest",
        verifier,
    )

    result = module.verify_promotion_acceptance_receipt_manifest_directory(root)

    assert result.root == root
    assert result.manifest is verified
    assert calls == [(receipt_directory, manifest_path)]


@pytest.mark.parametrize("missing", ["receipt-evidence", "acceptance-receipt-manifest.json"])
def test_rejects_missing_entries(tmp_path: Path, missing: str) -> None:
    root, receipt_directory, manifest_path = _fixture(tmp_path)
    target = receipt_directory if missing == "receipt-evidence" else manifest_path
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(ValueError, match="unexpected or missing entries"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)


def test_rejects_unexpected_entry(tmp_path: Path) -> None:
    root, _, _ = _fixture(tmp_path)
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing entries"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)


@pytest.mark.parametrize("entry_name", ["receipt-evidence", "acceptance-receipt-manifest.json"])
def test_rejects_symlinked_entries(tmp_path: Path, entry_name: str) -> None:
    root, receipt_directory, manifest_path = _fixture(tmp_path)
    entry = receipt_directory if entry_name == "receipt-evidence" else manifest_path
    if entry.is_dir():
        entry.rmdir()
        target = tmp_path / "receipt-target"
        target.mkdir()
    else:
        entry.unlink()
        target = tmp_path / "manifest-target.json"
        target.write_text("{}", encoding="utf-8")
    entry.symlink_to(target, target_is_directory=target.is_dir())

    with pytest.raises(ValueError, match="regular"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)


def test_rejects_wrong_entry_types(tmp_path: Path) -> None:
    root, receipt_directory, manifest_path = _fixture(tmp_path)
    receipt_directory.rmdir()
    receipt_directory.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt-evidence"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)

    receipt_directory.unlink()
    receipt_directory.mkdir()
    manifest_path.unlink()
    manifest_path.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)


def test_rejects_missing_or_symlinked_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_receipt_manifest_directory(missing)

    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "root"
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_receipt_manifest_directory(root)
