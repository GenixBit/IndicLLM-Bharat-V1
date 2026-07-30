from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_acceptance import (
    PromotionAcceptanceVerification,
    verify_promotion_acceptance,
)

_BUNDLE_NAME = "bundle"
_ACCEPTANCE_NAME = "acceptance.json"
_REQUIRED_ENTRIES = {_BUNDLE_NAME, _ACCEPTANCE_NAME}


@dataclass(frozen=True)
class PromotionAcceptanceDirectoryVerification:
    directory: Path
    acceptance: PromotionAcceptanceVerification


def verify_promotion_acceptance_directory(
    directory: Path,
) -> PromotionAcceptanceDirectoryVerification:
    """Verify a complete local tokenizer-promotion acceptance directory."""

    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("promotion acceptance directory must be a regular directory")

    entries = {entry.name for entry in directory.iterdir()}
    missing = sorted(_REQUIRED_ENTRIES - entries)
    unexpected = sorted(entries - _REQUIRED_ENTRIES)
    if missing:
        raise ValueError(f"promotion acceptance directory missing entries: {missing}")
    if unexpected:
        names = ", ".join(unexpected)
        raise ValueError(f"promotion acceptance directory has unexpected entries: {names}")

    bundle_directory = directory / _BUNDLE_NAME
    acceptance_path = directory / _ACCEPTANCE_NAME

    if bundle_directory.is_symlink() or not bundle_directory.is_dir():
        raise ValueError("promotion acceptance bundle must be a regular directory")
    if acceptance_path.is_symlink() or not acceptance_path.is_file():
        raise ValueError("promotion acceptance record must be a regular file")

    acceptance = verify_promotion_acceptance(bundle_directory, acceptance_path)
    return PromotionAcceptanceDirectoryVerification(directory, acceptance)
