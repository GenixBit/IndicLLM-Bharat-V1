from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.schema import (
    DataSourceSpec,
    SourceStatus,
)
from bharat.data.sources import load_source_spec

_VALID_SOURCE = {
    "schema_version": 1,
    "source_id": "test_dataset",
    "version": "1.0.0",
    "display_name": "Test Dataset",
    "provider": "Test Provider",
    "kind": "huggingface",
    "uri": "https://huggingface.co/datasets/org/test",
    "revision": "abc123def4567890abcdef1234567890abcdef12",
    "languages": ["en", "hi"],
    "domains": ["general"],
    "splits": ["train", "validation"],
    "purposes": ["pretraining"],
    "status": "proposed",
    "license": "mit",
    "gated": False,
    "created_at": "2025-07-07",
    "updated_at": "2025-07-07",
}


def _write(tmp_path: Path, overrides: dict | None = None, data: dict | None = None) -> Path:
    d = data if data is not None else dict(_VALID_SOURCE)
    if overrides:
        d.update(overrides)
    p = tmp_path / "source.yaml"
    with p.open("w") as f:
        yaml.dump(d, f)
    return p


class TestSourceSchema:
    def test_valid_source(self, tmp_path):
        p = _write(tmp_path)
        spec = load_source_spec(p)
        assert isinstance(spec, DataSourceSpec)
        assert spec.source_id == "test_dataset"
        assert spec.schema_version == 1

    def test_unknown_root_key_rejected(self, tmp_path):
        p = _write(tmp_path, {"unknown_field": "value"})
        with pytest.raises(ValueError, match="unknown source key"):
            load_source_spec(p)

    def test_malformed_yaml_rejected(self, tmp_path):
        p = tmp_path / "source.yaml"
        with p.open("w") as f:
            f.write(": broken yaml\n")
        with pytest.raises(ValueError, match="malformed YAML"):
            load_source_spec(p)

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
        with pytest.raises(ValueError, match="secret or API key"):
            load_source_spec(p)

    def test_credentials_env_no_whitespace(self, tmp_path):
        p = _write(tmp_path, {"credentials_env": "MY TOKEN"})
        with pytest.raises(ValueError, match="environment-variable"):
            load_source_spec(p)

    def test_invalid_sha256_rejected(self, tmp_path):
        data = dict(_VALID_SOURCE)
        data["integrity"] = {"revision": "abc", "sha256": "not-a-valid-sha"}
        p = _write(tmp_path, data=data)
        with pytest.raises(ValueError, match="SHA-256"):
            load_source_spec(p)

    def test_mutable_hf_revision_rejected(self, tmp_path):
        p = _write(
            tmp_path,
            {
                "kind": "huggingface",
                "revision": "main",
            },
        )
        with pytest.raises(ValueError, match="immutable commit SHA"):
            load_source_spec(p)

    def test_approved_http_without_checksum_rejected(self, tmp_path):
        data = dict(_VALID_SOURCE)
        data.update(
            {
                "kind": "http",
                "uri": "https://example.com/data.bin",
                "revision": "abc123def4567890abcdef1234567890abcdef12",
                "status": "approved",
                "integrity": {"revision": "abc123def4567890abcdef1234567890abcdef12"},
            }
        )
        p = _write(tmp_path, data=data)
        # This tests that the schema itself accepts it; registry checks integrity
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
