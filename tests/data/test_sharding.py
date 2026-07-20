from __future__ import annotations

import pytest

from bharat.data.sharding import ShardPlan, ShardPlanner


class TestShardPlanner:
    def test_single_shard(self):
        planner = ShardPlanner(dataset_id="ds", split="train", max_records_per_shard=10000)
        plans = planner.plan(total_records=500)
        assert len(plans) == 1
        assert plans[0].index == 0
        assert plans[0].expected_records == 500
        assert plans[0].shard_id == "ds_train_0000.jsonl"

    def test_multiple_shards_by_records(self):
        planner = ShardPlanner(dataset_id="ds", max_records_per_shard=100)
        plans = planner.plan(total_records=250)
        assert len(plans) == 3
        assert plans[0].expected_records == 84
        assert plans[1].expected_records == 83
        assert plans[2].expected_records == 83

    def test_shard_naming(self):
        planner = ShardPlanner(dataset_id="my_ds", split="val")
        plans = planner.plan(total_records=200)
        assert plans[0].shard_id == "my_ds_val_0000.jsonl"

    def test_zero_records(self):
        planner = ShardPlanner()
        plans = planner.plan(total_records=0)
        assert plans == ()

    def test_deterministic(self):
        planner = ShardPlanner(dataset_id="ds", max_records_per_shard=50)
        p1 = planner.plan(total_records=120)
        p2 = planner.plan(total_records=120)
        assert p1 == p2
        assert p1[0].planned_digest == p2[0].planned_digest

    def test_negative_records_rejected(self):
        planner = ShardPlanner()
        with pytest.raises(ValueError, match="total_records"):
            planner.plan(total_records=-1)

    def test_invalid_max_records(self):
        with pytest.raises(ValueError, match="max_records_per_shard"):
            ShardPlanner(max_records_per_shard=0)

    def test_invalid_max_bytes(self):
        with pytest.raises(ValueError, match="max_bytes_per_shard"):
            ShardPlanner(max_bytes_per_shard=-1)

    def test_byte_split_creates_more_shards(self):
        planner = ShardPlanner(
            dataset_id="ds",
            max_records_per_shard=10000,
            max_bytes_per_shard=100,
        )
        plans = planner.plan(total_records=500, total_bytes=1000)
        assert len(plans) > 1

    def test_shard_plan_attributes(self):
        plan = ShardPlan(
            shard_id="test_0000.jsonl",
            index=0,
            expected_records=100,
            expected_bytes=50000,
            planned_digest="abc123",
        )
        assert plan.shard_id == "test_0000.jsonl"
        assert plan.index == 0
        assert plan.expected_records == 100
        assert plan.expected_bytes == 50000
        assert plan.planned_digest == "abc123"
