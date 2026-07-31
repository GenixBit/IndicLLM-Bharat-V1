from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.tokenizer.promotion_acceptance_receipt_manifest_directory_digest import (
    PromotionAcceptanceReceiptManifestDirectoryDigestVerification,
    verify_promotion_acceptance_receipt_manifest_directory_digest,
)

_RECORD_SCHEMA = "tokenizer-promotion-acceptance-receipt-manifest-directory-digest-v1"
_REQUIRED_FIELDS = {"directory_sha256", "schema_version"}
_INVALID_FILE_MESSAGE = "promotion acceptance receipt manifest directory digest record must be a regular file"
_INVALID_JSON_MESSAGE = "promotion acceptance receipt manifest directory digest record must be valid UTF-8 JSON"
_INVALID_OBJECT_MESSAGE = "promotion acceptance receipt manifest directory digest record must be a JSON object"
_INVALID_FIELDS_MESSAGE = (
    "promotion acceptance receipt manifest directory digest record has unexpected or missing fields"
)
_INVALID_SCHEMA_MESSAGE = (
    "unsupported promotion acceptance receipt manifest directory digest record schema"
)


@dataclass(frozen=True)
class PromotionAcceptanceReceiptManifestDirectoryDigestRecordVerification:
    digest: PromotionAcceptanceReceiptManifestDirectoryDigestVerification
    record_path: Path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(_INVALID_JSON_MESSAGE) from exc
    if not isinstance(value, dict):
        raise ValueError(_INVALID_OBJECT_MESSAGE)
    return value


def verify_promotion_acceptance_receipt_manifest_directory_digest_record(
    root: Path,
    record_path: Path,
) -> PromotionAcceptanceReceiptManifestDirectoryDigestRecordVerification:
    """Verify a local digest record for completed acceptance-manifest evidence."""

    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError(_INVALID_FILE_MESSAGE)

    record = _load_json_object(record_path)
    if set(record) != _REQUIRED_FIELDS:
        raise ValueError(_INVALID_FIELDS_MESSAGE)
    if record["schema_version"] != _RECORD_SCHEMA:
        raise ValueError(_INVALID_SCHEMA_MESSAGE)

    verified = verify_promotion_acceptance_receipt_manifest_directory_digest(
        root,
        record["directory_sha256"],
    )
    return PromotionAcceptanceReceiptManifestDirectoryDigestRecordVerification(
        digest=verified,
        record_path=record_path,
    )
