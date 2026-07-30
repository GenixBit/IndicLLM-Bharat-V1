from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_bundle_directory import (
    PromotionBundleDirectoryVerification,
    verify_promotion_bundle_directory,
)

_ACCEPTANCE_SCHEMA = "tokenizer-promotion-acceptance-v1"
_REQUIRED_FIELDS = {
    "accepted",
    "operator",
    "receipt_sha256",
    "reviewer",
    "schema_version",
}


@dataclass(frozen=True)
class PromotionAcceptanceVerification:
    bundle: PromotionBundleDirectoryVerification
    operator: str
    reviewer: str


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("promotion acceptance must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("promotion acceptance must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_promotion_acceptance(
    bundle_directory: Path,
    acceptance_path: Path,
) -> PromotionAcceptanceVerification:
    """Verify a local acceptance record against one promotion bundle."""

    if acceptance_path.is_symlink() or not acceptance_path.is_file():
        raise ValueError("promotion acceptance must be a regular file")

    bundle = verify_promotion_bundle_directory(bundle_directory)
    acceptance = _load_json_object(acceptance_path)

    if set(acceptance) != _REQUIRED_FIELDS:
        raise ValueError("promotion acceptance has unexpected or missing fields")
    if acceptance["schema_version"] != _ACCEPTANCE_SCHEMA:
        raise ValueError("unsupported promotion acceptance schema")
    if acceptance["accepted"] is not True:
        raise ValueError("promotion acceptance must explicitly accept the bundle")

    receipt_path = bundle_directory / "receipt.json"
    if acceptance["receipt_sha256"] != _sha256(receipt_path):
        raise ValueError("promotion acceptance receipt_sha256 does not match")

    operator = acceptance["operator"]
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("promotion acceptance operator must be non-empty")
    if operator != bundle.receipt.operator:
        raise ValueError("promotion acceptance operator does not match receipt")

    reviewer = acceptance["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("promotion acceptance reviewer must be non-empty")
    if reviewer == operator:
        raise ValueError("promotion acceptance reviewer must differ from operator")

    return PromotionAcceptanceVerification(
        bundle=bundle,
        operator=operator,
        reviewer=reviewer,
    )
