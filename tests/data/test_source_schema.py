from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.schema import (
    DataSourceSpec,
    SourceStatus,
)
from bharat.data.sources import load_source_spec


def _write(tmp_path: Path, overrides: dict) -> Path:
    base = {
        "schema_version": 1,
        "source_id": "test_source",
        "version": "1.0.0",
        "display_name": "Test Source",
        "provider": "test_provider",
        "kind": "http",
        "uri": "https://example.com/data",
        "revision": "abc123def456abc123def456abc123def456abc1",
        "languages": ["en"],
        "domains": ["general"],
        "splits": ["train"],
        "purposes": ["pretraining"],
        "status": "proposed",
        "license": "cc-by-4.0",
        "created_at": "2025-01-01",
        "updated_at": "2025-06-01",
    }
    base.update(overrides)
    p = tmp_path / "source.yaml"
    with p.open("w") as f:
        yaml.dump(base, f)
    return p


class TestSourceSchema:
    def test_minimal_valid_proposed(self, tmp_path):
        p = _write(tmp_path, {})
        spec = load_source_spec(p)
        assert isinstance(spec, DataSourceSpec)
        assert spec.source_id == "test_source"
        assert spec.schema_version == 1

    def test_missing_fields_rejected(self, tmp_path):
        p = tmp_path / "source.yaml"
        with p.open("w") as f:
            yaml.dump({"schema_version": 1}, f)
        with pytest.raises((ValueError, TypeError)):
            load_source_spec(p)

    def test_invalid_slug_rejected(self, tmp_path):
        p = _write(tmp_path, {"source_id": "Invalid Slug!"})
        with pytest.raises(ValueError, match="source_id"):
            load_source_spec(p)

    def test_duplicate_languages_rejected(self, tmp_path):
        p = _write(tmp_path, {"languages": ["en", "en"]})
        with pytest.raises(ValueError, match="duplicate language"):
            load_source_spec(p)

    def test_invalid_date_rejected(self, tmp_path):
        p = _write(tmp_path, {"created_at": "not-a-date"})
        with pytest.raises(ValueError, match="invalid date"):
            load_source_spec(p)

    def test_uri_with_credentials_rejected(self, tmp_path):
        p = _write(tmp_path, {"uri": "https://user:pass@example.com/data"})
        with pytest.raises(ValueError, match="embedded credentials"):
            load_source_spec(p)

    def test_credentials_env_not_value(self, tmp_path):
        p = _write(tmp_path, {"credentials_env": "sk-abcdefghijklmnopqrstuvwxyz1234567890abc"})
        with pytest.raises(ValueError, match="environment-variable name"):
            load_source_spec(p)

    def test_credentials_env_invalid_format(self, tmp_path):
        p = _write(tmp_path, {"credentials_env": "MY VAR"})
        with pytest.raises(ValueError, match="environment-variable name"):
            load_source_spec(p)

    def test_credentials_env_valid(self, tmp_path):
        p = _write(tmp_path, {"credentials_env": "MY_API_KEY"})
        spec = load_source_spec(p)
        assert spec.credentials_env == "MY_API_KEY"

    def test_invalid_sha256_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "not-a-valid-sha",
                }
            },
        )
        with pytest.raises(ValueError, match="SHA-256"):
            load_source_spec(p)

    def test_mutable_hf_revision_rejected(self, tmp_path):
        p = _write(tmp_path, {"kind": "huggingface", "revision": "main"})
        with pytest.raises(ValueError, match="40-character"):
            load_source_spec(p)

    def test_invalid_hf_revision_rejected(self, tmp_path):
        p = _write(tmp_path, {"kind": "huggingface", "revision": "abc123"})
        with pytest.raises(ValueError, match="40-character"):
            load_source_spec(p)

        p2 = _write(
            tmp_path,
            {"kind": "huggingface", "revision": "ABCDEF0123456789ABCDEF0123456789ABCDEF01"},
        )
        with pytest.raises(ValueError, match="40-character"):
            load_source_spec(p2)

    def test_approved_http_without_checksum_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "kind": "http",
                "uri": "https://example.com/data.bin",
                "status": "approved",
                "integrity": {"revision": "abc123def456abc123def456abc123def456abc1"},
            },
        )
        spec = load_source_spec(p)
        assert spec.status == SourceStatus.APPROVED

    def test_rejected_source_with_reason(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "status": "rejected",
                "notes": "Contains PII that cannot be removed",
            },
        )
        spec = load_source_spec(p)
        assert spec.status == SourceStatus.REJECTED
        assert spec.notes == "Contains PII that cannot be removed"

    def test_deprecated_source(self, tmp_path):
        p = _write(tmp_path, {"status": "deprecated"})
        spec = load_source_spec(p)
        assert spec.status == SourceStatus.DEPRECATED

    def test_boolean_not_accepted_as_integer(self, tmp_path):
        p = _write(tmp_path, {"schema_version": True})
        with pytest.raises(TypeError, match=r"schema_version.*integer"):
            load_source_spec(p)

    def test_proposed_source_incomplete_metadata(self, tmp_path):
        p = _write(tmp_path, {"status": "proposed"})
        spec = load_source_spec(p)
        assert spec.status == SourceStatus.PROPOSED
        assert spec.integrity is None

    def test_version_validation(self, tmp_path):
        p = _write(tmp_path, {"version": "invalid-version"})
        with pytest.raises(ValueError, match="PEP 440"):
            load_source_spec(p)

    def test_version_10_gt_2(self, tmp_path):
        p = _write(tmp_path, {"version": "10.0.0"})
        spec = load_source_spec(p)
        assert spec.version == "10.0.0"

    def test_domain_non_string_rejected(self, tmp_path):
        p = _write(tmp_path, {"domains": [42]})
        with pytest.raises(TypeError, match="domain must be a string"):
            load_source_spec(p)

    def test_duplicate_domain_rejected(self, tmp_path):
        p = _write(tmp_path, {"domains": ["general", "general"]})
        with pytest.raises(ValueError, match="duplicate domain"):
            load_source_spec(p)

    def test_split_non_string_rejected(self, tmp_path):
        p = _write(tmp_path, {"splits": [42]})
        with pytest.raises(TypeError, match="split must be a string"):
            load_source_spec(p)

    def test_duplicate_split_rejected(self, tmp_path):
        p = _write(tmp_path, {"splits": ["train", "train"]})
        with pytest.raises(ValueError, match="duplicate split"):
            load_source_spec(p)

    def test_duplicate_purpose_rejected(self, tmp_path):
        p = _write(tmp_path, {"purposes": ["pretraining", "pretraining"]})
        with pytest.raises(ValueError, match="duplicate purpose"):
            load_source_spec(p)

    def test_duplicate_upstream_rejected(self, tmp_path):
        p = _write(tmp_path, {"upstream_sources": ["src1", "src1"]})
        with pytest.raises(ValueError, match="duplicate upstream source"):
            load_source_spec(p)

    def test_invalid_supersedes_type_rejected(self, tmp_path):
        p = _write(tmp_path, {"supersedes": 42})
        with pytest.raises(TypeError, match="supersedes must be a string or null"):
            load_source_spec(p)

    def test_invalid_notes_type_rejected(self, tmp_path):
        p = _write(tmp_path, {"notes": 42})
        with pytest.raises(TypeError, match="notes must be a string or null"):
            load_source_spec(p)

    def test_updated_before_created_rejected(self, tmp_path):
        p = _write(tmp_path, {"created_at": "2025-06-01", "updated_at": "2025-01-01"})
        with pytest.raises(ValueError, match=r"updated_at.*before created_at"):
            load_source_spec(p)

    def test_hf_revision_must_be_40_char_sha(self, tmp_path):
        p = _write(tmp_path, {"kind": "huggingface", "revision": "abc"})
        with pytest.raises(ValueError, match="40-character"):
            load_source_spec(p)

    def test_revision_integrity_mismatch_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "integrity": {
                    "revision": "0000000000000000000000000000000000000000",
                    "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                }
            },
        )
        with pytest.raises(ValueError, match="does not match"):
            load_source_spec(p)

    def test_manifest_uri_without_sha256_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                    "manifest_uri": "https://example.com/manifest",
                }
            },
        )
        with pytest.raises(ValueError, match="manifest_sha256 required"):
            load_source_spec(p)

    def test_manifest_sha256_without_uri_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                    "manifest_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                }
            },
        )
        with pytest.raises(ValueError, match="manifest_uri required"):
            load_source_spec(p)

    def test_dataset_card_url_must_be_https(self, tmp_path):
        p = _write(tmp_path, {"dataset_card_url": "http://example.com/card"})
        with pytest.raises(ValueError, match="https://"):
            load_source_spec(p)
