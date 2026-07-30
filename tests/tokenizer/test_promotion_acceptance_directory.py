from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_acceptance_directory as module


def _fixture(tmp_path: Path) -> Path:
    directory = tmp_path / "accepted-promotion"
    directory.mkdir()
    (directory / "bundle").mkdir()
    (directory / "acceptance.json").write_text("{}", encoding="utf-8")
    return directory


def _stub_verifier(monkeypatch: pytest.MonkeyPatch) -> object:
    verified = SimpleNamespace(operator="operator@example.com")
    monkeypatch.setattr(module, "verify_promotion_acceptance", lambda *_: verified)
    return verified


def test_verifies_complete_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _fixture(tmp_path)
    verified = _stub_verifier(monkeypatch)

    result = module.verify_promotion_acceptance_directory(directory)

    assert result.directory == directory
    assert result.acceptance is verified


def test_delegates_exact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _fixture(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def verifier(bundle: Path, acceptance: Path) -> object:
        calls.append((bundle, acceptance))
        return SimpleNamespace()

    monkeypatch.setattr(module, "verify_promotion_acceptance", verifier)

    module.verify_promotion_acceptance_directory(directory)

    assert calls == [(directory / "bundle", directory / "acceptance.json")]


@pytest.mark.parametrize("entry", ["bundle", "acceptance.json"])
def test_rejects_missing_required_entry(tmp_path: Path, entry: str) -> None:
    directory = _fixture(tmp_path)
    path = directory / entry
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()

    with pytest.raises(ValueError, match="missing entries"):
        module.verify_promotion_acceptance_directory(directory)


def test_rejects_unexpected_entry(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    (directory / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected entries"):
        module.verify_promotion_acceptance_directory(directory)


def test_rejects_missing_or_symlinked_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_directory(missing)

    target = _fixture(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        module.verify_promotion_acceptance_directory(link)


def test_rejects_invalid_bundle_type(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    bundle = directory / "bundle"
    bundle.rmdir()
    bundle.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle must be a regular directory"):
        module.verify_promotion_acceptance_directory(directory)


def test_rejects_symlinked_bundle(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    bundle = directory / "bundle"
    bundle.rmdir()
    target = tmp_path / "bundle-target"
    target.mkdir()
    bundle.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="bundle must be a regular directory"):
        module.verify_promotion_acceptance_directory(directory)


def test_rejects_invalid_acceptance_type(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    acceptance = directory / "acceptance.json"
    acceptance.unlink()
    acceptance.mkdir()

    with pytest.raises(ValueError, match="record must be a regular file"):
        module.verify_promotion_acceptance_directory(directory)


def test_rejects_symlinked_acceptance(tmp_path: Path) -> None:
    directory = _fixture(tmp_path)
    acceptance = directory / "acceptance.json"
    acceptance.unlink()
    target = tmp_path / "acceptance-target.json"
    target.write_text("{}", encoding="utf-8")
    acceptance.symlink_to(target)

    with pytest.raises(ValueError, match="record must be a regular file"):
        module.verify_promotion_acceptance_directory(directory)
