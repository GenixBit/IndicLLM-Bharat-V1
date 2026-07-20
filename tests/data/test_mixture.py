from __future__ import annotations

import pytest

from bharat.data.manifest import DatasetManifest
from bharat.data.mixture import MixtureConstraint, MixturePlanner


def _make_manifest(
    dataset_id: str,
    source_id: str,
    language: str,
    records: int,
    split: str = "train",
) -> DatasetManifest:
    return DatasetManifest(
        manifest_version="1.0",
        dataset_id=dataset_id,
        source_id=source_id,
        source_version="1.0.0",
        created_at="2026-07-20T12:00:00Z",
        license="cc-by-4.0",
        language=language,
        split=split,
        records=records,
        bytes_utf8=records * 500,
        sha256="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        processing_config_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
        registry_digest="def456abc123def456abc123def456abc123def456abc123def456abc123def456",
        policy_digest="123abc456def123abc456def123abc456def123abc456def123abc456def123abc",
    )


class TestMixturePlanner:
    def test_balanced_language_mix(self):
        manifests = [
            _make_manifest("en_ds", "en_source", "en", 10000),
            _make_manifest("hi_ds", "hi_source", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert len(plans) == 2
        assert plans[0].weight == plans[1].weight

    def test_source_cap_redistributes(self):
        manifests = [
            _make_manifest("big_ds", "big_source", "en", 90000),
            _make_manifest("small_ds", "small_source", "en", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.5,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        big_plan = [p for p in plans if p.source_id == "big_source"][0]
        assert "capped" in big_plan.note.lower()

    def test_deterministic(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000),
            _make_manifest("b", "src_b", "hi", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        p1 = planner.plan(manifests, constraint)
        p2 = planner.plan(manifests, constraint)
        assert p1 == p2

    def test_invalid_weights_rejected(self):
        manifests = [
            _make_manifest("ds", "src", "en", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.3, "hi": 0.3},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="sum to 1.0"):
            planner.plan(manifests, constraint)

    def test_empty_manifests_rejected(self):
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="At least one manifest"):
            planner.plan([], constraint)

    def test_min_record_threshold(self):
        manifests = [
            _make_manifest("ds", "src", "en", 100),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            min_record_threshold=1000,
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="below min threshold"):
            planner.plan(manifests, constraint)

    def test_negative_language_weight(self):
        manifests = [_make_manifest("ds", "src", "en", 5000)]
        constraint = MixtureConstraint(
            language_weights={"en": -0.1, "hi": 1.1},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="non-negative"):
            planner.plan(manifests, constraint)

    def test_single_language_mix(self):
        manifests = [
            _make_manifest("ds", "src_a", "en", 10000),
            _make_manifest("ds2", "src_b", "en", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert len(plans) == 2


class TestMixturePlannerHardening:
    def test_normalized_final_weights(self):
        manifests = [
            _make_manifest("en_ds", "src_a", "en", 10000),
            _make_manifest("hi_ds", "src_b", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.6, "hi": 0.4},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        total_weight = sum(p.weight for p in plans)
        assert abs(total_weight - 1.0) < 1e-9

    def test_source_cap_redistribution(self):
        manifests = [
            _make_manifest("big_ds", "big_source", "en", 90000),
            _make_manifest("small_ds", "small_source", "en", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.5,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert len(plans) == 2
        big_plan = [p for p in plans if p.source_id == "big_source"][0]
        assert "capped" in big_plan.note.lower()

    def test_impossible_source_cap_raises(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 60000),
            _make_manifest("b", "src_b", "hi", 60000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.4,
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="All sources exceed"):
            planner.plan(manifests, constraint)

    def test_domain_weights_validation(self):
        manifests = [
            _make_manifest("ds", "src", "en", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 0.6, "test": 0.3},
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="sum to 1.0"):
            planner.plan(manifests, constraint)

    def test_missing_language_zero_weight(self):
        manifests = [
            _make_manifest("ds", "src", "fr", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert plans[0].weight == 0.0
        assert plans[0].estimated_records == 0

    def test_domain_mapping(self):
        manifests = [
            _make_manifest("ds", "src_a", "en", 10000, split="train"),
            _make_manifest("ds2", "src_b", "hi", 10000, split="train"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"web": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(
            manifests,
            constraint,
            domain_mapping={"src_a": "web", "src_b": "web"},
        )
        assert all(p.domain == "web" for p in plans)

    def test_estimated_records_sum(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 10000),
            _make_manifest("b", "src_b", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        estimated_sum = sum(p.estimated_records for p in plans)
        assert estimated_sum > 0
