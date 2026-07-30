from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_acceptance_receipt import (
    PromotionAcceptanceReceiptVerification,
    verify_promotion_acceptance_receipt,
)

_ACCEPTANCE_DIRECTORY_NAME = "accepted-promotion"
_RECEIPT_NAME = "acceptance-receipt.json"
_REQUIRED_ENTRIES = {_ACCEPTANCE_DIRECTORY_NAME, _RECEIPT_NAME}


@dataclass(frozen=True)
class PromotionAcceptanceReceiptDirectoryVerification:
    directory: Path
    receipt: PromotionAcceptanceReceiptVerification


def verify_promotion_acceptance_receipt_directory(
    directory: Path,
) -> PromotionAcceptanceReceiptDirectoryVerification:
    """Verify a complete local directory containing accepted promotion evidence."""

    if directory.is_symlink() or not directory.is_dir():
        message = "promotion acceptance receipt directory must be a regular directory"
        raise ValueError(message)

    entries = {entry.name for entry in directory.iterdir()}
    missing = sorted(_REQUIRED_ENTRIES - entries)
    unexpected = sorted(entries - _REQUIRED_ENTRIES)
    if missing:
        message = f"promotion acceptance receipt directory missing entries: {missing}"
        raise ValueError(message)
    if unexpected:
        names = ", ".join(unexpected)
        message = (
            "promotion acceptance receipt directory has unexpected entries: "
            f"{names}"
        )
        raise ValueError(message)

    acceptance_directory = directory / _ACCEPTANCE_DIRECTORY_NAME
    receipt_path = directory / _RECEIPT_NAME

    if acceptance_directory.is_symlink() or not acceptance_directory.is_dir():
        raise ValueError("accepted promotion evidence must be a regular directory")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("promotion acceptance receipt must be a regular file")

    receipt = verify_promotion_acceptance_receipt(acceptance_directory, receipt_path)
    return PromotionAcceptanceReceiptDirectoryVerification(directory, receipt)
