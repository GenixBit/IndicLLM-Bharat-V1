from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import promotion_package_directory as module


def _complete_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    for name in ("manifest.json", "readiness.json", "decision.json"):
        (package / name).write_text("{}", encoding="utf-8")
    return package


def test_verifies_complete_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _complete_package(tmp_path)
    result = SimpleNamespace(manifest_sha256="abc")
    calls: list[tuple[Path, Path, Path]] = []

    def verify(manifest: Path, readiness: Path, decision: Path) -> SimpleNamespace:
        calls.append((manifest, readiness, decision))
        return result

    monkeypatch.setattr(module, "verify_promotion_package", verify)

    verified = module.verify_promotion_package_directory(package)

    assert verified.package is result
    assert verified.filenames == ("decision.json", "manifest.json", "readiness.json")
    assert calls == [
        (
            package / "manifest.json",
            package / "readiness.json",
            package / "decision.json",
        )
    ]


def test_rejects_missing_required_file(tmp_path: Path) -> None:
    package = _complete_package(tmp_path)
    (package / "decision.json").unlink()

    with pytest.raises(ValueError, match="missing required files: decision.json"):
        module.verify_promotion_package_directory(package)


def test_rejects_unexpected_entry(tmp_path: Path) -> None:
    package = _complete_package(tmp_path)
    (package / "notes.txt").write_text("not part of the package", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected entries: notes.txt"):
        module.verify_promotion_package_directory(package)


def test_rejects_symlinked_required_file(tmp_path: Path) -> None:
    package = _complete_package(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (package / "decision.json").unlink()
    (package / "decision.json").symlink_to(target)

    with pytest.raises(ValueError, match="must be a regular file: decision.json"):
        module.verify_promotion_package_directory(package)


def test_rejects_required_directory_entry(tmp_path: Path) -> None:
    package = _complete_package(tmp_path)
    (package / "readiness.json").unlink()
    (package / "readiness.json").mkdir()

    with pytest.raises(ValueError, match="must be a regular file: readiness.json"):
        module.verify_promotion_package_directory(package)


def test_rejects_non_directory_path(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="path must be a directory"):
        module.verify_promotion_package_directory(package)


def test_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory does not exist"):
        module.verify_promotion_package_directory(tmp_path / "missing")
