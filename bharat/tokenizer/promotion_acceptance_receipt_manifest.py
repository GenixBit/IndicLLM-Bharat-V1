from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_acceptance_receipt_directory import (
    PromotionAcceptanceReceiptDirectoryVerification,
    verify_promotion_acceptance_receipt_directory,
)

_MANIFEST_SCHEMA = "tokenizer-promotion-acceptance-receipt-manifest-v1"
_REQUIRED_FIELDS = {
    "operator",
    "receipt_sha256",
    "reviewer",
    "schema_version",
}
_RECEIPT_NAME = "acceptance-receipt.json"
_INVALID_JSON_MESSAGE = (
    "promotion acceptance receipt manifest must be valid UTF-8 JSON"
)
_INVALID_OBJECT_MESSAGE = (
    "promotion acceptance receipt manifest must be a JSON object"
)
_INVALID_FILE_MESSAGE = (
    "promotion acceptance receipt manifest must be a regular file"
)
_INVALID_FIELDS_MESSAGE = (
    "promotion acceptance receipt manifest has unexpected or missing fields"
)
_INVALID_SCHEMA_MESSAGE = (
    "unsupported promotion acceptance receipt manifest schema"
)
_INVALID_DIGEST_MESSAGE = (
    "promotion acceptance receipt manifest receipt_sha256 does not match"
)
_INVALID_OPERATOR_MESSAGE = (
    "promotion acceptance receipt manifest operator does not match"
)
_INVALID_REVIEWER_MESSAGE = (
    "promotion acceptance receipt manifest reviewer does not match"
)


@dataclass(frozen=True)
class PromotionAcceptanceReceiptManifestVerification:
    receipt_directory: PromotionAcceptanceReceiptDirectoryVerification
    operator: str
    reviewer: str


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(_INVALID_JSON_MESSAGE) from exc
    if not isinstance(value, dict):
        raise ValueError(_INVALID_OBJECT_MESSAGE)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_promotion_acceptance_receipt_manifest(
    receipt_directory: Path,
    manifest_path: Path,
) -> PromotionAcceptanceReceiptManifestVerification:
    """Verify an immutable local manifest for completed acceptance-receipt evidence."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(_INVALID_FILE_MESSAGE)

    verified = verify_promotion_acceptance_receipt_directory(receipt_directory)
    manifest = _load_json_object(manifest_path)

    if set(manifest) != _REQUIRED_FIELDS:
        raise ValueError(_INVALID_FIELDS_MESSAGE)
    if manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError(_INVALID_SCHEMA_MESSAGE)

    receipt_path = receipt_directory / _RECEIPT_NAME
    if manifest["receipt_sha256"] != _sha256(receipt_path):
        raise ValueError(_INVALID_DIGEST_MESSAGE)

    operator = manifest["operator"]
    reviewer = manifest["reviewer"]
    if operator != verified.receipt.operator:
        raise ValueError(_INVALID_OPERATOR_MESSAGE)
    if reviewer != verified.receipt.reviewer:
        raise ValueError(_INVALID_REVIEWER_MESSAGE)

    return PromotionAcceptanceReceiptManifestVerification(
        receipt_directory=verified,
        operator=operator,
        reviewer=reviewer,
    )
