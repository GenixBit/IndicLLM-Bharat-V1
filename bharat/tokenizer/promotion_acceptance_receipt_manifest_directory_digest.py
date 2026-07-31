from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

from bharat.tokenizer.promotion_acceptance_receipt_manifest_directory import (
    PromotionAcceptanceReceiptManifestDirectoryVerification,
    verify_promotion_acceptance_receipt_manifest_directory,
)

_INVALID_DIGEST_MESSAGE = "expected digest must be a lowercase SHA-256 hex string"
_DIGEST_MISMATCH_MESSAGE = (
    "promotion acceptance receipt manifest directory digest does not match"
)
_INVALID_ENTRY_MESSAGE = (
    "promotion acceptance receipt manifest directory contains a non-regular entry"
)


@dataclass(frozen=True)
class PromotionAcceptanceReceiptManifestDirectoryDigestVerification:
    directory: PromotionAcceptanceReceiptManifestDirectoryVerification
    sha256: str


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(_INVALID_DIGEST_MESSAGE)


def promotion_acceptance_receipt_manifest_directory_sha256(root: Path) -> str:
    """Return a deterministic digest for all regular files under a local evidence root."""

    hasher = hashlib.sha256()
    entries = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(_INVALID_ENTRY_MESSAGE)
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError(_INVALID_ENTRY_MESSAGE)

        relative_path = entry.relative_to(root).as_posix().encode("utf-8")
        content = entry.read_bytes()
        hasher.update(len(relative_path).to_bytes(8, "big"))
        hasher.update(relative_path)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)

    return hasher.hexdigest()


def verify_promotion_acceptance_receipt_manifest_directory_digest(
    root: Path,
    expected_sha256: str,
) -> PromotionAcceptanceReceiptManifestDirectoryDigestVerification:
    """Verify semantic evidence and its complete deterministic local directory digest."""

    _validate_digest(expected_sha256)
    verified = verify_promotion_acceptance_receipt_manifest_directory(root)
    actual_sha256 = promotion_acceptance_receipt_manifest_directory_sha256(root)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(_DIGEST_MISMATCH_MESSAGE)

    return PromotionAcceptanceReceiptManifestDirectoryDigestVerification(
        directory=verified,
        sha256=actual_sha256,
    )
