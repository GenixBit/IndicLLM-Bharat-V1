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
    domain: str | None = None,
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
        domain=domain if domain is not None else split,
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
        assert abs(sum(p.weight for p in plans) - 1.0) < 1e-9

    def test_source_cap_redistributes(self):
        manifests = [
            _make_manifest("big_ds", "big_source", "en", 90000),
            _make_manifest("small_ds", "small_source", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.8, "hi": 0.2},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.5,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        big_plan = [p for p in plans if p.source_id == "big_source"][0]
        assert abs(big_plan.weight - 0.5) < 1e-9
        assert "capped" in big_plan.note.lower()
        assert abs(sum(p.weight for p in plans) - 1.0) < 1e-9

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
        assert abs(sum(p.weight for p in plans) - 1.0) < 1e-9


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
            _make_manifest("small_ds", "small_source", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.8, "hi": 0.2},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.5,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert len(plans) == 2
        big_plan = [p for p in plans if p.source_id == "big_source"][0]
        assert "capped" in big_plan.note.lower()
        assert abs(big_plan.weight - 0.5) < 1e-9
        small_plan = [p for p in plans if p.source_id == "small_source"][0]
        assert abs(small_plan.weight - 0.5) < 1e-9

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

    def test_missing_language_excluded(self):
        manifests = [
            _make_manifest("en_ds", "valid_src", "en", 5000),
            _make_manifest("fr_ds", "missing_src", "fr", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        missing_plan = [p for p in plans if p.source_id == "missing_src"][0]
        assert missing_plan.weight == 0.0
        assert "excluded" in missing_plan.note.lower()

    def test_domain_mapping(self):
        manifests = [
            _make_manifest("ds", "src_a", "en", 10000, split="train", domain=""),
            _make_manifest("ds2", "src_b", "hi", 10000, split="train", domain=""),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"web": 1.0},
            max_pct_per_source=1.0,
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
        assert estimated_sum == 20000


class TestExplicitDomain:
    def test_explicit_domain_attribute(self):
        manifests = [
            _make_manifest("ds", "src", "en", 10000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"web": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert plans[0].domain == "web"

    def test_missing_domain_raises(self):
        m = DatasetManifest(
            manifest_version="1.0",
            dataset_id="ds",
            source_id="src",
            source_version="1.0.0",
            created_at="2026-07-20T12:00:00Z",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="",
            records=10000,
            bytes_utf8=5000000,
            sha256="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            processing_config_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abc1",
            registry_digest="def456abc123def456abc123def456abc123def456abc123def456abc123def456",
            policy_digest="123abc456def123abc456def123abc456def123abc456def123abc456def123abc",
        )
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="Cannot determine domain"):
            planner.plan([m], constraint)

    def test_split_fallback_when_enabled(self):
        manifests = [
            _make_manifest("ds", "src", "en", 10000, split="train", domain=""),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint, allow_split_fallback=True)
        assert plans[0].domain == "train"

    def test_domain_mapping_overrides_split(self):
        manifests = [
            _make_manifest("ds", "src", "en", 10000, split="train", domain=""),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"web": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(
            manifests,
            constraint,
            domain_mapping={"src": "web"},
        )
        assert plans[0].domain == "web"

    def test_manifest_domain_overrides_mapping(self):
        manifests = [
            _make_manifest("ds", "src", "en", 10000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"web": 0.5, "other": 0.5},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(
            manifests,
            constraint,
            domain_mapping={"src": "other"},
        )
        assert plans[0].domain == "web"


class TestSourceCap:
    def test_one_source_over_cap_gets_capped(self):
        manifests = [
            _make_manifest("big", "src_a", "en", 5000),
            _make_manifest("small", "src_b", "hi", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.9, "hi": 0.1},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.6,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        a = [p for p in plans if p.source_id == "src_a"][0]
        b = [p for p in plans if p.source_id == "src_b"][0]
        assert abs(a.weight - 0.6) < 1e-9
        assert abs(b.weight - 0.4) < 1e-9
        assert "capped" in a.note.lower()

    def test_multiple_capped_sources(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000),
            _make_manifest("b", "src_b", "hi", 5000),
            _make_manifest("c", "src_c", "te", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.4, "te": 0.1},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.35,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        assert len(plans) == 3
        capped = [p for p in plans if p.note and "capped" in p.note.lower()]
        assert len(capped) == 2
        assert abs(sum(p.weight for p in plans) - 1.0) < 1e-9

    def test_cap_redistribution_preserves_ratios(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000),
            _make_manifest("b", "src_b", "hi", 5000),
            _make_manifest("c", "src_c", "te", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.6, "hi": 0.3, "te": 0.1},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.5,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        b = [p for p in plans if p.source_id == "src_b"][0]
        c = [p for p in plans if p.source_id == "src_c"][0]
        ratio = b.weight / c.weight
        assert abs(ratio - 3.0) < 1e-6

    def test_deterministic_after_capping(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000),
            _make_manifest("b", "src_b", "hi", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.9, "hi": 0.1},
            domain_weights={"train": 1.0},
            max_pct_per_source=0.6,
        )
        planner = MixturePlanner()
        p1 = planner.plan(manifests, constraint)
        p2 = planner.plan(manifests, constraint)
        assert p1 == p2


class TestEstimatedRecords:
    def test_estimated_records_sum_exactly_target(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000, domain="web"),
            _make_manifest("b", "src_b", "hi", 5000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"web": 1.0},
            min_record_threshold=0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        est_sum = sum(p.estimated_records for p in plans)
        assert est_sum == 10000

    def test_deterministic_remainder_assignment(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"web": 1.0},
            min_record_threshold=0,
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        p1 = planner.plan(manifests, constraint)
        p2 = planner.plan(manifests, constraint)
        assert p1 == p2
        assert p1[0].estimated_records == 5000

    def test_target_records_smaller_than_sources(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000, domain="web"),
            _make_manifest("b", "src_b", "hi", 5000, domain="web"),
            _make_manifest("c", "src_c", "te", 5000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.3, "te": 0.2},
            domain_weights={"web": 1.0},
            min_record_threshold=0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint, target_records=2)
        est_sum = sum(p.estimated_records for p in plans)
        assert est_sum == 2
        assert all(0 <= p.estimated_records <= 2 for p in plans)

    def test_target_records_larger_than_total(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 5000, domain="web"),
            _make_manifest("b", "src_b", "hi", 5000, domain="web"),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.5, "hi": 0.5},
            domain_weights={"web": 1.0},
            min_record_threshold=0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint, target_records=200)
        est_sum = sum(p.estimated_records for p in plans)
        assert est_sum == 200

    def test_explicit_target_records(self):
        manifests = [
            _make_manifest("a", "src_a", "en", 10000),
            _make_manifest("b", "src_b", "hi", 10000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 0.6, "hi": 0.4},
            domain_weights={"train": 1.0},
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint, target_records=5000)
        est_sum = sum(p.estimated_records for p in plans)
        assert est_sum == 5000


class TestZeroWeight:
    def test_all_zero_weights_raises(self):
        manifests = [
            _make_manifest("en_ds", "src_a", "en", 5000),
            _make_manifest("hi_ds", "src_b", "hi", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"fr": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        with pytest.raises(ValueError, match="All source weights are zero"):
            planner.plan(manifests, constraint)

    def test_zero_weight_source_excluded(self):
        manifests = [
            _make_manifest("en_ds", "valid_src", "en", 5000),
            _make_manifest("fr_ds", "zero_src", "fr", 5000),
        ]
        constraint = MixtureConstraint(
            language_weights={"en": 1.0},
            domain_weights={"train": 1.0},
            max_pct_per_source=1.0,
        )
        planner = MixturePlanner()
        plans = planner.plan(manifests, constraint)
        zero_plan = [p for p in plans if p.source_id == "zero_src"][0]
        assert zero_plan.weight == 0.0
        assert zero_plan.estimated_records == 0
        assert "excluded" in zero_plan.note.lower()
