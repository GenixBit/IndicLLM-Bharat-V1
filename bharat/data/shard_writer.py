from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from bharat.data.manifest import ShardManifest
from bharat.data.records import ProcessedRecord


@dataclass(frozen=True)
class ShardWriterConfig:
    output_dir: str
    source_id: str
    split: str
    max_records_per_shard: int = 10000
    max_bytes_per_shard: int = 64 * 1024 * 1024


class ShardWriter:
    def __init__(self, config: ShardWriterConfig) -> None:
        self._config = config
        self._shard_index = 0
        self._manifests: list[ShardManifest] = []
        self._record_offset = 0
        self._out = Path(config.output_dir) / "shards"
        self._out.mkdir(parents=True, exist_ok=True)

    def _shard_name(self, index: int) -> str:
        return f"{self._config.source_id}.{self._config.split}.{index:05d}.jsonl"

    def write_shard(self, records: list[ProcessedRecord]) -> ShardManifest | None:
        accepted = [r for r in records if r.accepted]
        if not accepted:
            return None

        idx = self._shard_index
        name = self._shard_name(idx)
        tmp_name = f".tmp.{os.getpid()}.{name}"
        tmp_path = self._out / tmp_name
        final_path = self._out / name

        sha = hashlib.sha256()
        byte_count = 0
        with tmp_path.open("w", encoding="utf-8") as f:
            for r in accepted:
                line = json.dumps(r.to_dict(), ensure_ascii=False) + "\n"
                line_bytes = line.encode("utf-8")
                sha.update(line_bytes)
                byte_count += len(line_bytes)
                f.write(line)

        tmp_path.rename(final_path)

        record_start = self._record_offset
        record_end = self._record_offset + len(accepted)

        manifest = ShardManifest(
            shard_id=name,
            index=idx,
            record_start=record_start,
            record_end=record_end,
            bytes_utf8=byte_count,
            sha256=sha.hexdigest(),
        )
        self._manifests.append(manifest)
        self._shard_index += 1
        self._record_offset += len(accepted)
        return manifest

    @property
    def manifests(self) -> tuple[ShardManifest, ...]:
        return tuple(self._manifests)
