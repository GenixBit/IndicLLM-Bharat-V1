from __future__ import annotations

import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_acceptance_receipt_manifest_directory import (
    PromotionAcceptanceReceiptManifestDirectoryVerification,
    verify_promotion_acceptance_receipt_manifest_directory,
)

_CHANGED_DURING_VERIFICATION_MESSAGE = (
    "promotion acceptance receipt manifest directory changed during verification"
)
_DIGEST_MISMATCH_MESSAGE = "promotion acceptance receipt manifest directory digest does not match"
_INVALID_DIGEST_MESSAGE = "expected digest must be a lowercase SHA-256 hex string"
_INVALID_ENTRY_MESSAGE = (
    "promotion acceptance receipt manifest directory contains a non-regular entry"
)
_INVALID_ROOT_MESSAGE = (
    "promotion acceptance receipt manifest digest root must be a regular directory"
)
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class PromotionAcceptanceReceiptManifestDirectoryDigestVerification:
    directory: PromotionAcceptanceReceiptManifestDirectoryVerification
    sha256: str


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(_INVALID_DIGEST_MESSAGE)


def _require_regular_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(_INVALID_ROOT_MESSAGE) from exc
    if not stat.S_ISDIR(mode):
        raise ValueError(_INVALID_ROOT_MESSAGE)


def _update_with_streamed_content(hasher: Any, path: Path) -> None:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(_CHANGED_DURING_VERIFICATION_MESSAGE) from exc
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ValueError(_CHANGED_DURING_VERIFICATION_MESSAGE)
        hasher.update(opened.st_size.to_bytes(8, "big"))
        total = 0
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
    if total != opened.st_size:
        raise ValueError(_CHANGED_DURING_VERIFICATION_MESSAGE)


def promotion_acceptance_receipt_manifest_directory_sha256(root: Path) -> str:
    """Return a deterministic digest for all regular files under a local evidence root."""

    _require_regular_directory(root)
    hasher = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for entry in entries:
        try:
            mode = entry.lstat().st_mode
        except FileNotFoundError as exc:
            raise ValueError(_CHANGED_DURING_VERIFICATION_MESSAGE) from exc
        if stat.S_ISLNK(mode):
            raise ValueError(_INVALID_ENTRY_MESSAGE)
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(_INVALID_ENTRY_MESSAGE)

        relative_path = entry.relative_to(root).as_posix().encode("utf-8")
        hasher.update(len(relative_path).to_bytes(8, "big"))
        hasher.update(relative_path)
        _update_with_streamed_content(hasher, entry)

    return hasher.hexdigest()


def verify_promotion_acceptance_receipt_manifest_directory_digest(
    root: Path,
    expected_sha256: str,
) -> PromotionAcceptanceReceiptManifestDirectoryDigestVerification:
    """Verify semantic evidence and its complete deterministic local directory digest."""

    _validate_digest(expected_sha256)
    _require_regular_directory(root)
    digest_before = promotion_acceptance_receipt_manifest_directory_sha256(root)
    verified = verify_promotion_acceptance_receipt_manifest_directory(root)
    digest_after = promotion_acceptance_receipt_manifest_directory_sha256(root)
    if not hmac.compare_digest(digest_before, digest_after):
        raise ValueError(_CHANGED_DURING_VERIFICATION_MESSAGE)
    if not hmac.compare_digest(digest_after, expected_sha256):
        raise ValueError(_DIGEST_MISMATCH_MESSAGE)

    return PromotionAcceptanceReceiptManifestDirectoryDigestVerification(
        directory=verified,
        sha256=digest_after,
    )
