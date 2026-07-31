from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_acceptance_receipt_manifest import (
    PromotionAcceptanceReceiptManifestVerification,
    verify_promotion_acceptance_receipt_manifest,
)

_RECEIPT_DIRECTORY_NAME = "receipt-evidence"
_MANIFEST_NAME = "acceptance-receipt-manifest.json"
_REQUIRED_ENTRIES = {_RECEIPT_DIRECTORY_NAME, _MANIFEST_NAME}
_INVALID_ROOT_MESSAGE = "promotion acceptance receipt manifest directory must be a regular directory"
_INVALID_ENTRIES_MESSAGE = (
    "promotion acceptance receipt manifest directory has unexpected or missing entries"
)
_INVALID_RECEIPT_DIRECTORY_MESSAGE = "receipt-evidence must be a regular directory"
_INVALID_MANIFEST_MESSAGE = "acceptance-receipt-manifest.json must be a regular file"


@dataclass(frozen=True)
class PromotionAcceptanceReceiptManifestDirectoryVerification:
    root: Path
    manifest: PromotionAcceptanceReceiptManifestVerification


def verify_promotion_acceptance_receipt_manifest_directory(
    root: Path,
) -> PromotionAcceptanceReceiptManifestDirectoryVerification:
    """Verify the exact local filesystem envelope for receipt manifest evidence."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(_INVALID_ROOT_MESSAGE)

    entries = {entry.name for entry in root.iterdir()}
    if entries != _REQUIRED_ENTRIES:
        raise ValueError(_INVALID_ENTRIES_MESSAGE)

    receipt_directory = root / _RECEIPT_DIRECTORY_NAME
    manifest_path = root / _MANIFEST_NAME
    if receipt_directory.is_symlink() or not receipt_directory.is_dir():
        raise ValueError(_INVALID_RECEIPT_DIRECTORY_MESSAGE)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(_INVALID_MANIFEST_MESSAGE)

    verified = verify_promotion_acceptance_receipt_manifest(
        receipt_directory,
        manifest_path,
    )
    return PromotionAcceptanceReceiptManifestDirectoryVerification(
        root=root,
        manifest=verified,
    )
