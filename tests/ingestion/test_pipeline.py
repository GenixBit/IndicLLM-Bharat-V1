from __future__ import annotations

from pathlib import Path

from bharat.ingestion.pipeline import ContinuousIngestionPipeline, DocumentRecord


class TestIngestionPipeline:
    def test_ingest_file_and_deduplication(self, tmp_path: Path):
        pipeline = ContinuousIngestionPipeline(state_dir=tmp_path / "state")

        test_file = tmp_path / "sample.md"
        test_file.write_text(
            "# Knowledge Base Document\nThis is a sovereign test document on Indian AI.",
            encoding="utf-8",
        )

        doc1 = pipeline.ingest_file(test_file)
        assert isinstance(doc1, DocumentRecord)
        assert doc1.version == 1
        assert len(doc1.chunks) > 0

        # Duplicate ingestion should be skipped
        doc2 = pipeline.ingest_file(test_file)
        assert doc2 is None
