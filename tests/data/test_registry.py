from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.registry import DataRegistry
from bharat.data.schema import UsagePurpose


def _make_source(tmp_path: Path, filename: str, overrides: dict | None = None) -> Path:
    data = {
        "schema_version": 1,
        "source_id": "ds1",
        "version": "1.0.0",
        "display_name": "Dataset 1",
        "provider": "test",
        "kind": "http",
        "uri": "https://example.com/ds1",
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
    if overrides:
        data.update(overrides)
    p = tmp_path / f"{filename}.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


def _policy_path(tmp_path: Path, policy_dir: Path | None = None) -> Path:
    data = {
        "schema_version": 1,
        "default_decision": "deny",
        "licenses": [
            {
                "identifier": "cc-by-4.0",
                "name": "CC BY 4.0",
                "decision": "allow",
                "evidence_url": "https://example.com/cc-by-4.0",
                "verified_at": "2025-06-01",
                "verified_by": "project_team",
                "commercial_use_allowed": True,
                "model_training_allowed": True,
                "redistribution_allowed": True,
                "attribution_required": True,
                "share_alike": False,
            }
        ],
    }
    target_dir = policy_dir or tmp_path
    p = target_dir / "license_policy.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


class TestRegistry:
    def test_empty_registry(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert registry.list_all() == ()

    def test_load_single_source(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1")
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        sources = registry.list_all()
        assert len(sources) == 1
        assert sources[0].source_id == "ds1"

    def test_duplicate_source_id_version_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1_a")
        _make_source(registry_dir, "ds1_b")
        with pytest.raises(ValueError, match="Duplicate source/version pair"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_proposed_plus_proposed_duplicate_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "a",
            {
                "source_id": "a",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        _make_source(
            registry_dir,
            "b",
            {
                "source_id": "b",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        with pytest.raises(ValueError, match="Duplicate active source"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_proposed_plus_approved_duplicate_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        integrity = {
            "revision": "abc123def456abc123def456abc123def456abc1",
            "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
        }
        _make_source(
            registry_dir,
            "a",
            {
                "source_id": "a",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        _make_source(
            registry_dir,
            "b",
            {
                "source_id": "b",
                "status": "approved",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
                "integrity": integrity,
            },
        )
        with pytest.raises(ValueError, match="Duplicate active source"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_plus_approved_duplicate_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        integrity = {
            "revision": "abc123def456abc123def456abc123def456abc1",
            "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
        }
        _make_source(
            registry_dir,
            "a",
            {
                "source_id": "a",
                "status": "approved",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
                "integrity": integrity,
            },
        )
        _make_source(
            registry_dir,
            "b",
            {
                "source_id": "b",
                "status": "approved",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
                "integrity": integrity,
            },
        )
        with pytest.raises(ValueError, match="Duplicate active source"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_deprecated_historical_duplicate_allowed(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "a",
            {
                "source_id": "a",
                "status": "deprecated",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        _make_source(
            registry_dir,
            "b",
            {
                "source_id": "b",
                "status": "deprecated",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert len(registry.list_all()) == 2

    def test_rejected_historical_duplicate_allowed(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "a",
            {
                "source_id": "a",
                "status": "rejected",
                "notes": "not suitable",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        _make_source(
            registry_dir,
            "b",
            {
                "source_id": "b",
                "status": "rejected",
                "notes": "not suitable",
                "uri": "https://example.com/data",
                "revision": "abc123def456abc123def456abc123def456abc1",
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert len(registry.list_all()) == 2

    def test_version_semantics_10_gt_2(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds_v2",
            {
                "source_id": "ds",
                "version": "2.0.0",
                "uri": "https://example.com/ds/v2",
                "revision": "2222222222222222222222222222222222222222",
                "created_at": "2025-01-02",
                "updated_at": "2025-06-02",
            },
        )
        _make_source(
            registry_dir,
            "ds_v10",
            {
                "source_id": "ds",
                "version": "10.0.0",
                "uri": "https://example.com/ds/v10",
                "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "created_at": "2025-01-03",
                "updated_at": "2025-06-03",
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        spec = registry.get("ds")
        assert spec is not None
        assert spec.version == "10.0.0"

    def test_version_prerelease_ordering(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds_alpha",
            {
                "source_id": "ds",
                "version": "1.0.0-alpha",
                "uri": "https://example.com/ds/alpha",
                "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "created_at": "2025-01-02",
                "updated_at": "2025-06-02",
            },
        )
        _make_source(
            registry_dir,
            "ds_ga",
            {
                "source_id": "ds",
                "version": "1.0.0",
                "uri": "https://example.com/ds/ga",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "created_at": "2025-01-03",
                "updated_at": "2025-06-03",
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        spec = registry.get("ds")
        assert spec is not None
        assert spec.version == "1.0.0"

    def test_notes_change_alters_digest(self, tmp_path):
        d1 = tmp_path / "r1"
        d1_src = d1 / "sources"
        d1_src.mkdir(parents=True)
        _policy_path(d1)
        _make_source(d1_src, "ds1", {"notes": "first version"})
        dig1 = DataRegistry.load(d1_src, policy_path=d1 / "license_policy.yaml").digest()

        d2 = tmp_path / "r2"
        d2_src = d2 / "sources"
        d2_src.mkdir(parents=True)
        _policy_path(d2)
        _make_source(d2_src, "ds1", {"notes": "second version"})
        dig2 = DataRegistry.load(d2_src, policy_path=d2 / "license_policy.yaml").digest()

        assert dig1 != dig2

    def test_policy_change_alters_digest(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1")
        dig1 = DataRegistry.load(
            registry_dir, policy_path=tmp_path / "license_policy.yaml"
        ).digest()

        diff_policy = tmp_path / "alt_policy.yaml"
        data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [
                {
                    "identifier": "cc-by-4.0",
                    "name": "CC BY 4.0",
                    "decision": "allow",
                    "evidence_url": "https://example.com/cc-by-4.0",
                    "verified_at": "2025-07-01",
                    "verified_by": "other_team",
                    "commercial_use_allowed": True,
                    "model_training_allowed": True,
                    "redistribution_allowed": True,
                    "attribution_required": True,
                    "share_alike": False,
                }
            ],
        }
        with diff_policy.open("w") as f:
            yaml.dump(data, f)
        dig2 = DataRegistry.load(registry_dir, policy_path=diff_policy).digest()
        assert dig1 != dig2

    def test_digest_deterministic(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1")
        _make_source(
            registry_dir,
            "ds2",
            {
                "source_id": "ds2",
                "uri": "https://example.com/ds2",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            },
        )
        dig1 = DataRegistry.load(
            registry_dir, policy_path=tmp_path / "license_policy.yaml"
        ).digest()
        dig2 = DataRegistry.load(
            registry_dir, policy_path=tmp_path / "license_policy.yaml"
        ).digest()
        assert dig1 == dig2
        assert isinstance(dig1, str)
        assert len(dig1) == 64

    def test_supersession_source_at_version(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1")
        _make_source(
            registry_dir,
            "ds2",
            {
                "source_id": "ds2",
                "version": "2.0.0",
                "uri": "https://example.com/ds2",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "supersedes": "ds1@1.0.0",
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert len(registry.list_all()) == 2

    def test_supersession_self_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1", {"supersedes": "ds1@1.0.0"})
        with pytest.raises(ValueError, match="cannot supersede itself"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_supersession_unknown_target_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1", {"supersedes": "nonexistent@1.0.0"})
        with pytest.raises(ValueError, match="does not exist"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_supersession_cycle_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds_a",
            {
                "source_id": "ds_a",
                "uri": "https://example.com/a",
                "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "supersedes": "ds_b@1.0.0",
            },
        )
        _make_source(
            registry_dir,
            "ds_b",
            {
                "source_id": "ds_b",
                "version": "1.0.0",
                "uri": "https://example.com/b",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "supersedes": "ds_a@1.0.0",
            },
        )
        with pytest.raises(ValueError, match="cycle"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_hf_sha256_required_for_approval(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "kind": "huggingface",
                "uri": "https://huggingface.co/datasets/org/ds1",
                "revision": "abc123def456abc123def456abc123def456abc1",
                "status": "approved",
                "integrity": {"revision": "abc123def456abc123def456abc123def456abc1"},
            },
        )
        with pytest.raises(ValueError, match="SHA-256 required"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_http_sha256_required_for_approval(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "integrity": {"revision": "abc123def456abc123def456abc123def456abc1"},
            },
        )
        with pytest.raises(ValueError, match="SHA-256 checksum required"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_with_deny_license_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        p = tmp_path / "license_policy.yaml"
        with p.open("w") as f:
            yaml.dump(
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [
                        {
                            "identifier": "cc-by-4.0",
                            "name": "CC BY 4.0",
                            "decision": "deny",
                        }
                    ],
                },
                f,
            )
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        with pytest.raises(ValueError, match="must be 'allow'"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_with_review_license_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        p = tmp_path / "license_policy.yaml"
        with p.open("w") as f:
            yaml.dump(
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [
                        {
                            "identifier": "cc-by-4.0",
                            "name": "CC BY 4.0",
                            "decision": "review",
                        }
                    ],
                },
                f,
            )
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        with pytest.raises(ValueError, match="must be 'allow'"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_without_license_in_policy_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "license": "unknown-license",
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        with pytest.raises(ValueError, match="not found in policy"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_missing_integrity_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1", {"status": "approved"})
        with pytest.raises(ValueError, match="integrity record required"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_allow_licence_missing_evidence_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        p = tmp_path / "license_policy.yaml"
        with p.open("w") as f:
            yaml.dump(
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [
                        {
                            "identifier": "cc-by-4.0",
                            "name": "CC BY 4.0",
                            "decision": "allow",
                        }
                    ],
                },
                f,
            )
        with pytest.raises(ValueError, match="evidence_url"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_revision_integrity_mismatch_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "revision": "abc123def456abc123def456abc123def456abc1",
                "integrity": {
                    "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        with pytest.raises(ValueError, match="revision"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_for_validates_licence(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "source_id": "ds1",
                "status": "approved",
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        _make_source(
            registry_dir,
            "ds2",
            {
                "source_id": "ds2",
                "status": "approved",
                "uri": "https://example.com/ds2",
                "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "integrity": {
                    "revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        approved = registry.approved_for(UsagePurpose.PRETRAINING)
        assert len(approved) == 2
        assert {s.source_id for s in approved} == {"ds1", "ds2"}

    def test_to_snapshot_includes_policy_digest(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _policy_path(tmp_path)
        _make_source(registry_dir, "ds1")
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        snapshot = registry.to_snapshot()
        assert snapshot["schema_version"] == 1
        assert isinstance(snapshot["registry_digest"], str)
        assert len(snapshot["registry_digest"]) == 64
        assert isinstance(snapshot["policy_digest"], str)
        assert len(snapshot["policy_digest"]) == 64
        assert isinstance(snapshot["policy"], dict)
        assert isinstance(snapshot["sources"], list)
        assert len(snapshot["sources"]) == 1
