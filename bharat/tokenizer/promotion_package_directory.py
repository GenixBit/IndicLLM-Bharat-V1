from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_package import (
    PromotionPackageVerification,
    verify_promotion_package,
)

_DECISION_FILENAME = "decision.json"
_MANIFEST_FILENAME = "manifest.json"
_READINESS_FILENAME = "readiness.json"
_REQUIRED_FILES = (
    _DECISION_FILENAME,
    _MANIFEST_FILENAME,
    _READINESS_FILENAME,
)


@dataclass(frozen=True)
class PromotionPackageDirectoryVerification:
    package: PromotionPackageVerification
    filenames: tuple[str, ...]


def _require_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"required package file is missing: {path.name}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"package entry must be a regular file: {path.name}")


def verify_promotion_package_directory(
    directory: Path,
) -> PromotionPackageDirectoryVerification:
    """Verify one complete, local promotion package directory without mutation."""

    try:
        directory_mode = directory.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError("promotion package directory does not exist") from exc
    if not stat.S_ISDIR(directory_mode):
        raise ValueError("promotion package path must be a directory")

    entries = {entry.name: entry for entry in directory.iterdir()}
    actual = set(entries)
    required = set(_REQUIRED_FILES)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing:
        raise ValueError(
            f"promotion package is missing required files: {', '.join(missing)}"
        )
    if unexpected:
        raise ValueError(
            f"promotion package contains unexpected entries: {', '.join(unexpected)}"
        )

    for name in _REQUIRED_FILES:
        _require_regular_file(entries[name])

    package = verify_promotion_package(
        entries[_MANIFEST_FILENAME],
        entries[_READINESS_FILENAME],
        entries[_DECISION_FILENAME],
    )
    return PromotionPackageDirectoryVerification(
        package=package,
        filenames=_REQUIRED_FILES,
    )
