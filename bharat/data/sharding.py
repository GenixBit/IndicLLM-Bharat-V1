from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    index: int
    expected_records: int
    expected_bytes: int = 0
    planned_digest: str = ""


class ShardPlanner:
    def __init__(
        self,
        dataset_id: str = "dataset",
        split: str = "train",
        max_records_per_shard: int = 10000,
        max_bytes_per_shard: int = 100 * 1024 * 1024,
    ) -> None:
        if max_records_per_shard < 1:
            raise ValueError("max_records_per_shard must be >= 1")
        if max_bytes_per_shard < 1:
            raise ValueError("max_bytes_per_shard must be >= 1")
        self._dataset_id = dataset_id
        self._split = split
        self._max_records = max_records_per_shard
        self._max_bytes = max_bytes_per_shard

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def split(self) -> str:
        return self._split

    @property
    def max_records_per_shard(self) -> int:
        return self._max_records

    @property
    def max_bytes_per_shard(self) -> int:
        return self._max_bytes

    def _shard_name(self, index: int) -> str:
        return f"{self._dataset_id}_{self._split}_{index:04d}.jsonl"

    def plan(
        self,
        total_records: int,
        total_bytes: int = 0,
    ) -> tuple[ShardPlan, ...]:
        if total_records < 0:
            raise ValueError("total_records must be >= 0")
        if total_bytes < 0:
            raise ValueError("total_bytes must be >= 0")
        if total_records == 0:
            return ()

        records_per_shard = self._max_records
        num_shards_by_records = (total_records + records_per_shard - 1) // records_per_shard

        num_shards_by_bytes = 1
        if total_bytes > 0:
            num_shards_by_bytes = (total_bytes + self._max_bytes - 1) // self._max_bytes

        num_shards = max(num_shards_by_records, num_shards_by_bytes)

        base_records = total_records // num_shards
        extra_records = total_records % num_shards

        base_bytes = (total_bytes // num_shards) if total_bytes > 0 else 0
        extra_bytes = (total_bytes % num_shards) if total_bytes > 0 else 0

        plans: list[ShardPlan] = []
        record_offset = 0
        byte_offset = 0
        for idx in range(num_shards):
            shard_records = base_records + (1 if idx < extra_records else 0)
            shard_bytes = base_bytes + (1 if idx < extra_bytes else 0)
            shard_id = self._shard_name(idx)
            planned_digest = hashlib.sha256(
                f"{shard_id}:{shard_records}:{shard_bytes}".encode("utf-8")
            ).hexdigest()[:16]
            plans.append(
                ShardPlan(
                    shard_id=shard_id,
                    index=idx,
                    expected_records=shard_records,
                    expected_bytes=shard_bytes,
                    planned_digest=planned_digest,
                )
            )
            record_offset += shard_records
            byte_offset += shard_bytes

        return tuple(plans)
