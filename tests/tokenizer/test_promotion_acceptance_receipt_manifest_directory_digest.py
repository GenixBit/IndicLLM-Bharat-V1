from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_receipt_manifest_directory_digest as module


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "manifest-evidence"
    receipt_directory = root / "receipt-evidence"
    acceptance_directory = receipt_directory / "accepted-promotion"
    acceptance_directory.mkdir(parents=True)
    (acceptance_directory / "decision.json").write_text("accepted", encoding="utf-8")
    (receipt_directory / "acceptance-receipt.json").write_text("receipt", encoding="utf-8")
    (root / "acceptance-receipt-manifest.json").write_text("manifest", encoding="utf-8")
    return root


def test_verifies_exact_directory_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    verified = SimpleNamespace(root=root)
    calls: list[Path] = []

    def verifier(path: Path) -> object:
        calls.append(path)
        return verified

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        verifier,
    )
    expected = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    result = module.verify_promotion_acceptance_receipt_manifest_directory_digest(
        root,
        expected,
    )

    assert result.directory is verified
    assert result.sha256 == expected
    assert calls == [root]


def test_digest_is_deterministic_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    (first / "b.txt").write_text("two", encoding="utf-8")
    (first / "a.txt").write_text("one", encoding="utf-8")
    (second / "a.txt").write_text("one", encoding="utf-8")
    (second / "b.txt").write_text("two", encoding="utf-8")

    first_digest = module.promotion_acceptance_receipt_manifest_directory_sha256(first)
    second_digest = module.promotion_acceptance_receipt_manifest_directory_sha256(second)

    assert first_digest == second_digest


def test_digest_changes_with_path_or_content(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "evidence.json"
    path.write_text("one", encoding="utf-8")
    original = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    path.write_text("two", encoding="utf-8")
    changed_content = module.promotion_acceptance_receipt_manifest_directory_sha256(root)
    path.rename(root / "renamed.json")
    changed_path = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    assert changed_content != original
    assert changed_path != changed_content


@pytest.mark.parametrize(
    "value",
    ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64],
)
def test_rejects_invalid_expected_digest(tmp_path: Path, value: str) -> None:
    root = _fixture(tmp_path)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, value)


def test_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        lambda path: SimpleNamespace(root=path),
    )

    with pytest.raises(ValueError, match="digest does not match"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(
            root,
            "0" * 64,
        )


def test_rejects_symlinked_entries(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    (root / "evidence.txt").symlink_to(target)

    with pytest.raises(ValueError, match="non-regular entry"):
        module.promotion_acceptance_receipt_manifest_directory_sha256(root)
