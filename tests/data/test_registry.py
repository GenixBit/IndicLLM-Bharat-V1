from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.registry import DataRegistry
from bharat.data.schema import SourceStatus, UsagePurpose

_MIT_LICENSE = {
    "identifier": "mit",
    "name": "MIT License",
    "decision": "allow",
    "evidence_url": "https://opensource.org/licenses/MIT",
    "verified_at": "2025-07-01",
    "verified_by": "project_team",
    "commercial_use_allowed": True,
    "model_training_allowed": True,
    "redistribution_allowed": True,
}


def _make_policy(
    tmp_path: Path,
    licenses: list | None = None,
    default_decision: str = "deny",
) -> Path:
    data = {
        "schema_version": 1,
        "default_decision": default_decision,
        "licenses": licenses or [],
    }
    p = tmp_path / "license_policy.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


def _make_source(
    registry_dir: Path,
    source_id: str,
    overrides: dict | None = None,
    filename: str | None = None,
) -> Path:
    data: dict = {
        "schema_version": 1,
        "source_id": source_id,
        "version": "1.0.0",
        "display_name": source_id.title(),
        "provider": "Test",
        "kind": "huggingface",
        "uri": f"https://huggingface.co/datasets/org/{source_id}",
        "revision": f"{source_id}_abc123def4567890abcdef1234567890abcdef12",
        "languages": ["en"],
        "domains": ["general"],
        "splits": ["train"],
        "purposes": ["pretraining"],
        "status": "proposed",
        "license": "mit",
        "gated": False,
        "created_at": "2025-07-07",
        "updated_at": "2025-07-07",
    }
    if overrides:
        data.update(overrides)
    fname = filename or f"{source_id}.yaml"
    p = registry_dir / fname
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


def _make_mit_policy(tmp_path: Path) -> Path:
    return _make_policy(tmp_path, licenses=[_MIT_LICENSE])


class TestRegistry:
    def test_empty_registry(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert registry.list_all() == ()

    def test_duplicate_source_version_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1", filename="ds1_a.yaml")
        _make_source(registry_dir, "ds1", filename="ds1_b.yaml")
        with pytest.raises(ValueError, match="Duplicate"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_deterministic_ordering(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "b_source")
        _make_source(registry_dir, "a_source")
        _make_source(
            registry_dir,
            "a_source",
            {
                "version": "2.0.0",
                "revision": "a2_abc123def4567890abcdef1234567890abcdef12",
            },
            filename="a_source_v2.yaml",
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        sources = registry.list_all()
        assert len(sources) == 3
        assert sources[0].source_id == "a_source"
        assert sources[0].version == "1.0.0"
        assert sources[1].source_id == "a_source"
        assert sources[1].version == "2.0.0"
        assert sources[2].source_id == "b_source"

    def test_deterministic_digest(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1")
        _make_source(registry_dir, "ds2")
        d1 = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml").digest()
        d2 = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml").digest()
        assert d1 == d2
        assert isinstance(d1, str)
        assert len(d1) == 64

    def test_digest_changes_when_metadata_changes(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1")
        d1 = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml").digest()
        _make_source(
            registry_dir,
            "ds1",
            {
                "version": "2.0.0",
                "revision": "ds1_v2_abc123def4567890abcdef1234567890abcdef",
            },
            filename="ds1_v2.yaml",
        )
        d2 = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml").digest()
        assert d1 != d2

    def test_lookup_latest_version(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds")
        _make_source(
            registry_dir,
            "ds",
            {
                "version": "2.0.0",
                "revision": "ds_v2_abc123def4567890abcdef1234567890abcdef",
            },
            filename="ds_v2.yaml",
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        spec = registry.get("ds")
        assert spec is not None
        assert spec.version == "2.0.0"

    def test_lookup_specific_version(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds")
        _make_source(
            registry_dir,
            "ds",
            {
                "version": "2.0.0",
                "revision": "ds_v2_abc123def4567890abcdef1234567890abcdef",
            },
            filename="ds_v2.yaml",
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        spec = registry.get("ds", version="1.0.0")
        assert spec is not None
        assert spec.version == "1.0.0"

    def test_status_filter(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_mit_policy(tmp_path)
        _make_source(registry_dir, "proposed_ds")
        _make_source(
            registry_dir,
            "approved_ds",
            {
                "status": "approved",
                "revision": "app_v1_abc123def4567890abcdef1234567890abcdef",
                "integrity": {
                    "revision": "app_v1_abc123def4567890abcdef1234567890abcdef",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        proposed = registry.filter(status=SourceStatus.PROPOSED)
        assert len(proposed) == 1
        assert proposed[0].source_id == "proposed_ds"

    def test_purpose_filter(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "pretrain_ds")
        _make_source(registry_dir, "eval_ds", {"purposes": ["evaluation"]})
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        eval_sources = registry.filter(purpose=UsagePurpose.EVALUATION)
        assert len(eval_sources) == 1
        assert eval_sources[0].source_id == "eval_ds"

    def test_deprecated_excluded_from_approved_for(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_mit_policy(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "revision": "ds1v1_abc123def4567890abcdef1234567890abcdef12",
                "integrity": {
                    "revision": "ds1v1_abc123def4567890abcdef1234567890abcdef12",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        _make_source(
            registry_dir,
            "ds2",
            {
                "status": "deprecated",
                "revision": "ds2v1_abc123def4567890abcdef1234567890abcdef12",
                "integrity": {
                    "revision": "ds2v1_abc123def4567890abcdef1234567890abcdef12",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        approved = registry.approved_for(UsagePurpose.PRETRAINING)
        assert len(approved) == 1
        assert approved[0].source_id == "ds1"

    def test_missing_supersession_reference(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1", {"supersedes": "nonexistent"})
        with pytest.raises(ValueError, match="supersedes"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_self_supersession_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1", {"supersedes": "ds1"})
        with pytest.raises(ValueError, match="cannot supersede itself"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_supersession_cycle_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds_a", {"supersedes": "ds_b"})
        _make_source(registry_dir, "ds_b", {"supersedes": "ds_a"})
        with pytest.raises(ValueError, match="cycle"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_without_integrity_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_mit_policy(tmp_path)
        _make_source(registry_dir, "ds1", {"status": "approved"})
        with pytest.raises(ValueError, match="integrity record required"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_approved_without_licence_evidence_rejected(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(
            registry_dir,
            "ds1",
            {
                "status": "approved",
                "license": "unlisted_license",
                "revision": "ul_v1_abc123def4567890abcdef1234567890abcdef12",
                "integrity": {
                    "revision": "ul_v1_abc123def4567890abcdef1234567890abcdef12",
                    "sha256": "2687f86ed6784b8a5fca36e6c468e12aa44dc3c7e8137e3160d1a95079bdcd02",
                },
            },
        )
        with pytest.raises(ValueError, match="decision"):
            DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")

    def test_examples_directory_ignored(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        _make_policy(tmp_path)
        _make_source(registry_dir, "ds1")
        template_data = {
            "schema_version": 1,
            "source_id": "template",
            "version": "1.0.0",
            "display_name": "Template",
            "provider": "T",
            "kind": "http",
            "uri": "https://example.com/data",
            "revision": "tpl_v1_abc123def4567890abcdef1234567890abcdef",
            "languages": ["en"],
            "domains": ["general"],
            "splits": ["train"],
            "purposes": ["pretraining"],
            "status": "proposed",
            "license": "mit",
            "gated": False,
            "created_at": "2025-07-07",
            "updated_at": "2025-07-07",
        }
        p = examples_dir / "template.yaml"
        with p.open("w") as f:
            yaml.dump(template_data, f)
        registry = DataRegistry.load(registry_dir, policy_path=tmp_path / "license_policy.yaml")
        assert len(registry.list_all()) == 1
