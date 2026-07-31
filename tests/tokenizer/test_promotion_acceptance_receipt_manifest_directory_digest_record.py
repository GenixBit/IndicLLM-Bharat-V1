from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import (
    promotion_acceptance_receipt_manifest_directory_digest_record as module,
)


def _write_record(path: Path, digest: str = "a" * 64) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "tokenizer-promotion-acceptance-receipt-manifest-directory-digest-v1"
                ),
                "directory_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_verifies_digest_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    record_path = tmp_path / "digest-record.json"
    _write_record(record_path)
    verified = SimpleNamespace(sha256="a" * 64)
    calls: list[tuple[Path, str]] = []

    def verifier(path: Path, digest: str) -> object:
        calls.append((path, digest))
        return verified

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory_digest",
        verifier,
    )

    result = module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
        root,
        record_path,
    )

    assert result.digest is verified
    assert result.record_path == record_path
    assert calls == [(root, "a" * 64)]


def test_rejects_missing_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a regular file"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            tmp_path,
            tmp_path / "missing.json",
        )


def test_rejects_symlinked_record(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_record(target)
    link = tmp_path / "record.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must be a regular file"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            tmp_path,
            link,
        )


@pytest.mark.parametrize("content", ["not-json", "[]"])
def test_rejects_invalid_json_object(tmp_path: Path, content: str) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="valid UTF-8 JSON|must be a JSON object"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            tmp_path,
            record_path,
        )


def test_rejects_unexpected_fields(tmp_path: Path) -> None:
    record_path = tmp_path / "record.json"
    _write_record(record_path)
    value = json.loads(record_path.read_text(encoding="utf-8"))
    value["unexpected"] = True
    record_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            tmp_path,
            record_path,
        )


def test_rejects_unsupported_schema(tmp_path: Path) -> None:
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps({"schema_version": "wrong", "directory_sha256": "a" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            tmp_path,
            record_path,
        )


def test_delegates_digest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    record_path = tmp_path / "record.json"
    _write_record(record_path, "invalid")

    def verifier(path: Path, digest: str) -> object:
        raise ValueError("expected digest must be a lowercase SHA-256 hex string")

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory_digest",
        verifier,
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest_record(
            root,
            record_path,
        )
