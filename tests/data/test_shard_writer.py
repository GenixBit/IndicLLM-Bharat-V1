from __future__ import annotations

import hashlib
import json

from bharat.data.records import ProcessedRecord
from bharat.data.shard_writer import ShardWriter, ShardWriterConfig


def _make_record(text: str, accepted: bool = True) -> ProcessedRecord:
    return ProcessedRecord(
        record_id=hashlib.sha256(text.encode()).hexdigest()[:16],
        text=text,
        language="en",
        quality_score=0.95,
        source_path="/f.txt",
        line_number=1,
        processing_reasons=(),
        accepted=accepted,
    )


class TestShardWriter:
    def test_writes_single_shard(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="test",
            split="train",
            max_records_per_shard=100,
        )
        writer = ShardWriter(config)
        records = [_make_record(f"record_{i}") for i in range(5)]
        manifest = writer.write_shard(records)
        assert manifest.index == 0
        shard_path = tmp_path / "shards" / manifest.shard_id
        assert shard_path.exists()
        lines = shard_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_shard_sha256_matches(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(output_dir=out, source_id="src", split="train")
        writer = ShardWriter(config)
        records = [_make_record("hello")]
        manifest = writer.write_shard(records)
        shard_path = tmp_path / "shards" / manifest.shard_id
        actual_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        assert manifest.sha256 == actual_sha

    def test_deterministic_shard_names(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_records_per_shard=2,
        )
        writer = ShardWriter(config)
        writer.write_shard([_make_record("a"), _make_record("b")])
        writer.write_shard([_make_record("c")])
        assert len(writer.manifests) == 2
        assert writer.manifests[0].shard_id == "src.train.00000.jsonl"
        assert writer.manifests[1].shard_id == "src.train.00001.jsonl"

    def test_only_accepted_records_written(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(output_dir=out, source_id="src", split="train")
        writer = ShardWriter(config)
        records = [
            _make_record("good", accepted=True),
            _make_record("bad", accepted=False),
        ]
        manifest = writer.write_shard(records)
        shard_path = tmp_path / "shards" / manifest.shard_id
        lines = shard_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["text"] == "good"

    def test_all_rejected_returns_none(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(output_dir=out, source_id="src", split="train")
        writer = ShardWriter(config)
        records = [_make_record("bad", accepted=False)]
        result = writer.write_shard(records)
        assert result is None
        assert len(writer.manifests) == 0

    def test_record_offsets_contiguous(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_records_per_shard=3,
        )
        writer = ShardWriter(config)
        writer.write_shard([_make_record(f"r{i}") for i in range(3)])
        writer.write_shard([_make_record(f"s{i}") for i in range(2)])
        m0, m1 = writer.manifests
        assert m0.record_start == 0
        assert m0.record_end == 3
        assert m1.record_start == 3
        assert m1.record_end == 5
