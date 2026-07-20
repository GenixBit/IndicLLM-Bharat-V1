from __future__ import annotations

import json

import pytest

from bharat.data.manifest import (
    DatasetManifest,
    ShardManifest,
    create_manifest,
    digest_processing_config,
)

_SAMPLE_SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_SAMPLE_DIGEST = "abc123def456abc123def456abc123def456abc123def456abc123def456abc1"


def _make_minimal_manifest(**kwargs) -> DatasetManifest:
    defaults = dict(
        dataset_id="test_ds",
        source_id="test_source",
        source_version="1.0.0",
        license="cc-by-4.0",
        language="en",
        split="train",
        records=100,
        bytes_utf8=50000,
        sha256=_SAMPLE_SHA256,
        processing_config_digest=_SAMPLE_DIGEST,
        registry_digest=_SAMPLE_DIGEST,
        policy_digest=_SAMPLE_DIGEST,
    )
    defaults.update(kwargs)
    return create_manifest(**defaults)


class TestShardManifest:
    def test_minimal(self):
        s = ShardManifest(shard_id="shard_0000", index=0, record_start=0, record_end=100)
        assert s.shard_id == "shard_0000"
        assert s.index == 0
        assert s.record_start == 0
        assert s.record_end == 100

    def test_roundtrip_dict(self):
        s = ShardManifest(
            shard_id="shard_0000",
            index=0,
            record_start=0,
            record_end=100,
            bytes_utf8=50000,
            sha256=_SAMPLE_SHA256,
            created_at="2026-07-20T12:00:00Z",
        )
        d = s.to_dict()
        s2 = ShardManifest.from_dict(d)
        assert s == s2

    def test_missing_required_fields(self):
        with pytest.raises(ValueError, match="shard_id must be a non-empty string"):
            ShardManifest.from_dict({"index": 0, "record_start": 0, "record_end": 100})

    def test_negative_index(self):
        with pytest.raises(ValueError, match="index must be a non-negative integer"):
            ShardManifest.from_dict(
                {"shard_id": "s", "index": -1, "record_start": 0, "record_end": 10}
            )

    def test_record_end_before_start(self):
        with pytest.raises(ValueError, match="record_end"):
            ShardManifest.from_dict(
                {"shard_id": "s", "index": 0, "record_start": 50, "record_end": 10}
            )

    def test_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown shard key"):
            ShardManifest.from_dict(
                {
                    "shard_id": "s",
                    "index": 0,
                    "record_start": 0,
                    "record_end": 10,
                    "unknown_field": "bad",
                }
            )


class TestDatasetManifest:
    def test_minimal_creation(self):
        m = _make_minimal_manifest()
        assert m.dataset_id == "test_ds"
        assert m.records == 100
        assert m.manifest_version == "1.0"
        assert m.is_valid()

    def test_roundtrip_dict(self):
        m = _make_minimal_manifest()
        d = m.to_dict()
        m2 = DatasetManifest.from_dict(d)
        assert m == m2

    def test_deterministic_digest(self):
        m1 = _make_minimal_manifest()
        m2 = _make_minimal_manifest()
        assert m1.digest() == m2.digest()

    def test_digest_changes_with_records(self):
        m1 = _make_minimal_manifest(records=100)
        m2 = _make_minimal_manifest(records=200)
        assert m1.digest() != m2.digest()

    def test_schema_validation_rejects_bad_version(self):
        data = _make_minimal_manifest().to_dict()
        data["manifest_version"] = "0.5"
        with pytest.raises(ValueError, match="Unsupported manifest_version"):
            DatasetManifest.from_dict(data)

    def test_validation_rejects_negative_records(self):
        m = _make_minimal_manifest(records=-1)
        errors = m.validate()
        assert any("records must be non-negative" in e for e in errors)

    def test_shard_record_sum_mismatch(self):
        shards = (
            ShardManifest("s_0000", 0, 0, 50),
            ShardManifest("s_0001", 1, 50, 90),
        )
        with pytest.raises(ValueError, match="Shard record sum"):
            _make_minimal_manifest(records=100, shards=shards)

    def test_json_serialization(self):
        m = _make_minimal_manifest()
        raw = json.dumps(m.to_dict())
        m2 = DatasetManifest.from_dict(json.loads(raw))
        assert m == m2

    def test_unknown_root_keys(self):
        data = _make_minimal_manifest().to_dict()
        data["bad_key"] = "value"
        with pytest.raises(ValueError, match="Unknown manifest key"):
            DatasetManifest.from_dict(data)

    def test_empty_shards_valid(self):
        m = _make_minimal_manifest()
        assert m.shards == ()
        assert m.is_valid()

    def test_duplicate_shard_index(self):
        shards = (
            ShardManifest("s_0000", 0, 0, 50),
            ShardManifest("s_0001", 0, 50, 100),
        )
        m = _make_minimal_manifest(records=100, shards=shards)
        errors = m.validate()
        assert any("Duplicate shard index" in e for e in errors)

    def test_create_manifest_sets_created_at(self):
        m = create_manifest(
            dataset_id="ds",
            source_id="src",
            source_version="1.0",
            license="mit",
            language="en",
            split="train",
            records=10,
            bytes_utf8=1000,
            sha256=_SAMPLE_SHA256,
            processing_config_digest=_SAMPLE_DIGEST,
            registry_digest=_SAMPLE_DIGEST,
            policy_digest=_SAMPLE_DIGEST,
        )
        assert m.created_at
        assert m.dataset_id == "ds"

    def test_validate_no_empty_sha256(self):
        m = _make_minimal_manifest(sha256="")
        assert not m.is_valid()


class TestDigestProcessingConfig:
    def test_digest_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            threshold: float = 0.8
            flag: bool = True

        d1 = digest_processing_config(FakeConfig())
        d2 = digest_processing_config(FakeConfig())
        assert d1 == d2

    def test_digest_dict(self):
        d1 = digest_processing_config({"a": 1, "b": 2})
        d2 = digest_processing_config({"b": 2, "a": 1})
        assert d1 == d2

    def test_digest_differs(self):
        d1 = digest_processing_config({"a": 1})
        d2 = digest_processing_config({"a": 2})
        assert d1 != d2
