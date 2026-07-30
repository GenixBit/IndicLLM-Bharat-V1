from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_package_directory import (
    PromotionPackageDirectoryVerification,
    verify_promotion_package_directory,
)

_RECEIPT_SCHEMA = "tokenizer-promotion-receipt-v1"
_REQUIRED_FIELDS = {
    "decision_sha256",
    "manifest_sha256",
    "operator",
    "readiness_sha256",
    "schema_version",
}


@dataclass(frozen=True)
class PromotionReceiptVerification:
    package: PromotionPackageDirectoryVerification
    operator: str


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("promotion receipt must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("promotion receipt must be a JSON object")
    return value


def verify_promotion_receipt(
    package_directory: Path,
    receipt_path: Path,
) -> PromotionReceiptVerification:
    """Verify a local immutable receipt against one promotion package."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("promotion receipt must be a regular file")

    package = verify_promotion_package_directory(package_directory)
    receipt = _load_json_object(receipt_path)

    if set(receipt) != _REQUIRED_FIELDS:
        raise ValueError("promotion receipt has unexpected or missing fields")
    if receipt["schema_version"] != _RECEIPT_SCHEMA:
        raise ValueError("unsupported promotion receipt schema")

    verified = package.package
    expected_digests = {
        "manifest_sha256": verified.manifest_sha256,
        "readiness_sha256": verified.readiness_sha256,
        "decision_sha256": verified.decision_sha256,
    }
    for field, expected in expected_digests.items():
        if receipt[field] != expected:
            raise ValueError(f"promotion receipt {field} does not match")

    operator = receipt["operator"]
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("promotion receipt operator must be non-empty")
    if operator != verified.operator:
        raise ValueError("promotion receipt operator does not match decision")

    return PromotionReceiptVerification(package=package, operator=operator)
