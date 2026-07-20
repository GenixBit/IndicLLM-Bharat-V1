from __future__ import annotations

import json

import pytest

from bharat.data.records import ProcessedRecord, RawRecord


class TestRawRecord:
    def test_deterministic_record_id(self):
        r1 = RawRecord(source_path="/a/b.txt", line_number=1, text="hello")
        r2 = RawRecord(source_path="/a/b.txt", line_number=1, text="hello")
        assert r1.record_id == r2.record_id

    def test_different_line_different_id(self):
        r1 = RawRecord(source_path="/a/b.txt", line_number=1, text="hello")
        r2 = RawRecord(source_path="/a/b.txt", line_number=2, text="hello")
        assert r1.record_id != r2.record_id

    def test_to_dict(self):
        r = RawRecord(source_path="/f.txt", line_number=10, text="data")
        d = r.to_dict()
        assert d["record_id"] == r.record_id
        assert d["text"] == "data"
        assert d["source_path"] == "/f.txt"
        assert d["line_number"] == 10

    def test_to_dict_with_metadata(self):
        r = RawRecord(
            source_path="/f.txt", line_number=1, text="x", metadata={"key": "val"}
        )
        d = r.to_dict()
        assert d["metadata"] == {"key": "val"}


class TestProcessedRecord:
    def test_minimal(self):
        r = ProcessedRecord(
            record_id="abc123",
            text="hello",
            language="en",
            quality_score=0.95,
            source_path="/f.txt",
            line_number=1,
        )
        assert r.accepted
        assert r.processing_reasons == ()

    def test_to_dict(self):
        r = ProcessedRecord(
            record_id="id1",
            text="hello",
            language="en",
            quality_score=0.9,
            source_path="/f.txt",
            line_number=1,
            processing_reasons=("unsafe:hate",),
            accepted=False,
        )
        d = r.to_dict()
        assert d["record_id"] == "id1"
        assert d["processing_reasons"] == ["unsafe:hate"]
        assert d["accepted"] is False

    def test_digest_deterministic(self):
        r1 = ProcessedRecord(
            record_id="id1",
            text="hello",
            language="en",
            quality_score=0.9,
            source_path="/f.txt",
            line_number=1,
        )
        r2 = ProcessedRecord(
            record_id="id1",
            text="hello",
            language="en",
            quality_score=0.9,
            source_path="/f.txt",
            line_number=1,
        )
        assert r1.digest() == r2.digest()

    def test_digest_changes_with_field(self):
        r1 = ProcessedRecord(
            record_id="id1",
            text="hello",
            language="en",
            quality_score=0.9,
            source_path="/f.txt",
            line_number=1,
        )
        r2 = ProcessedRecord(
            record_id="id2",
            text="hello",
            language="en",
            quality_score=0.9,
            source_path="/f.txt",
            line_number=1,
        )
        assert r1.digest() != r2.digest()

    def test_json_roundtrip(self):
        r = ProcessedRecord(
            record_id="id1",
            text="hello",
            language="en",
            quality_score=0.95,
            source_path="/f.txt",
            line_number=5,
            processing_reasons=("quality:too_short",),
            accepted=False,
        )
        raw = json.dumps(r.to_dict())
        loaded = json.loads(raw)
        assert loaded["record_id"] == "id1"
        assert loaded["processing_reasons"] == ["quality:too_short"]
