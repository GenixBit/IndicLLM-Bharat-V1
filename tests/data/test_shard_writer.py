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
        manifests = writer.write_shard(records)
        assert len(manifests) == 1
        assert manifests[0].index == 0
        shard_path = tmp_path / "shards" / manifests[0].shard_id
        assert shard_path.exists()
        lines = shard_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_shard_sha256_matches(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(output_dir=out, source_id="src", split="train")
        writer = ShardWriter(config)
        records = [_make_record("hello")]
        manifests = writer.write_shard(records)
        assert len(manifests) == 1
        shard_path = tmp_path / "shards" / manifests[0].shard_id
        actual_sha = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        assert manifests[0].sha256 == actual_sha

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
        manifests = writer.write_shard(records)
        shard_path = tmp_path / "shards" / manifests[0].shard_id
        lines = shard_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["text"] == "good"

    def test_all_rejected_returns_empty(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(output_dir=out, source_id="src", split="train")
        writer = ShardWriter(config)
        records = [_make_record("bad", accepted=False)]
        result = writer.write_shard(records)
        assert result == []
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

    def test_byte_split_creates_multiple_shards(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_records_per_shard=100,
            max_bytes_per_shard=350,
        )
        writer = ShardWriter(config)
        records = [_make_record("x" * 100) for _ in range(5)]
        manifests = writer.write_shard(records)
        assert len(manifests) >= 2
        for m in manifests:
            assert m.bytes_utf8 <= 350
        # bytes limit triggered splitting (not record limit since max_records=100)
        assert len(manifests) > 1

    def test_record_split_still_works(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_records_per_shard=2,
            max_bytes_per_shard=10_000_000,
        )
        writer = ShardWriter(config)
        records = [_make_record(f"rec_{i}") for i in range(5)]
        manifests = writer.write_shard(records)
        assert len(manifests) == 3

    def test_oversized_record_raises(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_bytes_per_shard=10,
        )
        writer = ShardWriter(config)
        huge = _make_record("A" * 1000)
        import pytest

        with pytest.raises(ValueError, match="exceeds max_bytes_per_shard"):
            writer.write_shard([huge])

    def test_shard_byte_counts_match_files(self, tmp_path):
        out = str(tmp_path)
        config = ShardWriterConfig(
            output_dir=out,
            source_id="src",
            split="train",
            max_bytes_per_shard=500,
        )
        writer = ShardWriter(config)
        records = [_make_record("x" * 200) for _ in range(10)]
        manifests = writer.write_shard(records)
        for m in manifests:
            shard_path = tmp_path / "shards" / m.shard_id
            assert shard_path.stat().st_size == m.bytes_utf8

    def test_invalid_max_records_raises(self):
        import pytest

        with pytest.raises(ValueError, match="max_records_per_shard"):
            ShardWriter(
                ShardWriterConfig(
                    output_dir="/tmp",
                    source_id="src",
                    split="train",
                    max_records_per_shard=0,
                )
            )

    def test_invalid_max_bytes_raises(self):
        import pytest

        with pytest.raises(ValueError, match="max_bytes_per_shard"):
            ShardWriter(
                ShardWriterConfig(
                    output_dir="/tmp",
                    source_id="src",
                    split="train",
                    max_bytes_per_shard=0,
                )
            )
