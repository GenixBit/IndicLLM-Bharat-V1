from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_acceptance_directory import (
    PromotionAcceptanceDirectoryVerification,
    verify_promotion_acceptance_directory,
)

_RECEIPT_SCHEMA = "tokenizer-promotion-acceptance-receipt-v1"
_REQUIRED_FIELDS = {
    "acceptance_sha256",
    "operator",
    "reviewer",
    "schema_version",
}


@dataclass(frozen=True)
class PromotionAcceptanceReceiptVerification:
    acceptance_directory: PromotionAcceptanceDirectoryVerification
    operator: str
    reviewer: str


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("promotion acceptance receipt must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("promotion acceptance receipt must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_promotion_acceptance_receipt(
    acceptance_directory: Path,
    receipt_path: Path,
) -> PromotionAcceptanceReceiptVerification:
    """Verify an immutable local receipt for one accepted promotion directory."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("promotion acceptance receipt must be a regular file")

    verified = verify_promotion_acceptance_directory(acceptance_directory)
    receipt = _load_json_object(receipt_path)

    if set(receipt) != _REQUIRED_FIELDS:
        raise ValueError("promotion acceptance receipt has unexpected or missing fields")
    if receipt["schema_version"] != _RECEIPT_SCHEMA:
        raise ValueError("unsupported promotion acceptance receipt schema")

    acceptance_path = acceptance_directory / "acceptance.json"
    if receipt["acceptance_sha256"] != _sha256(acceptance_path):
        raise ValueError("promotion acceptance receipt acceptance_sha256 does not match")

    operator = receipt["operator"]
    reviewer = receipt["reviewer"]
    if operator != verified.acceptance.operator:
        raise ValueError("promotion acceptance receipt operator does not match")
    if reviewer != verified.acceptance.reviewer:
        raise ValueError("promotion acceptance receipt reviewer does not match")

    return PromotionAcceptanceReceiptVerification(verified, operator, reviewer)
