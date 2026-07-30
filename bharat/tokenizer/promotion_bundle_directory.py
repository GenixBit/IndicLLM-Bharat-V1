from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_receipt import (
    PromotionReceiptVerification,
    verify_promotion_receipt,
)

_PACKAGE_DIRECTORY = "package"
_RECEIPT_FILE = "receipt.json"
_REQUIRED_ENTRIES = {_PACKAGE_DIRECTORY, _RECEIPT_FILE}


@dataclass(frozen=True)
class PromotionBundleDirectoryVerification:
    bundle_directory: Path
    receipt: PromotionReceiptVerification


def verify_promotion_bundle_directory(
    bundle_directory: Path,
) -> PromotionBundleDirectoryVerification:
    """Verify one complete local promotion bundle directory."""

    if bundle_directory.is_symlink() or not bundle_directory.is_dir():
        raise ValueError("promotion bundle must be a regular directory")

    entries = {entry.name for entry in bundle_directory.iterdir()}
    missing = _REQUIRED_ENTRIES - entries
    unexpected = entries - _REQUIRED_ENTRIES
    if missing:
        raise ValueError(
            "promotion bundle is missing required entries: " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "promotion bundle has unexpected entries: "
            + ", ".join(sorted(unexpected))
        )

    package_directory = bundle_directory / _PACKAGE_DIRECTORY
    receipt_path = bundle_directory / _RECEIPT_FILE
    if package_directory.is_symlink() or not package_directory.is_dir():
        raise ValueError("promotion bundle package must be a regular directory")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("promotion bundle receipt must be a regular file")

    receipt = verify_promotion_receipt(package_directory, receipt_path)
    return PromotionBundleDirectoryVerification(
        bundle_directory=bundle_directory,
        receipt=receipt,
    )
