from __future__ import annotations

import json

from bharat.data.preparation import LocalPreparer, PreparationConfig

REAL_TEXT = (
    "The Indian education system has undergone significant changes in recent decades.\n"
    "With the introduction of the National Education Policy 2020, there is a renewed focus on holistic learning.\n"
    "This policy emphasizes critical thinking, experiential learning, and multidisciplinary approaches.\n"
    "It aims to transform India into a vibrant knowledge society and global knowledge superpower.\n"
    "The policy also focuses on early childhood care and education, foundational literacy, and numeracy."
)


class TestPreparation:
    def test_dry_run_writes_no_shards(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=str(tmp_path / "out"),
            dry_run=True,
        )
        preparer = LocalPreparer(config)
        manifest, report = preparer.prepare(str(f))
        assert report.total_records >= 1
        assert report.shard_count == 0
        assert manifest is not None

    def test_real_run_writes_shards(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=False,
        )
        preparer = LocalPreparer(config)
        manifest, report = preparer.prepare(str(f))
        shard_dir = tmp_path / "out" / "shards"
        assert shard_dir.exists()
        shard_files = list(shard_dir.glob("*.jsonl"))
        assert len(shard_files) >= 1
        assert report.shard_count >= 1

    def test_manifest_matches_accepted_records(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=False,
        )
        preparer = LocalPreparer(config)
        manifest, report = preparer.prepare(str(f))
        assert manifest.records == report.accepted_records
        assert manifest.records > 0

    def test_manifest_json_roundtrip(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=False,
        )
        preparer = LocalPreparer(config)
        manifest, _ = preparer.prepare(str(f))
        manifest_path = tmp_path / "out" / "manifest.json"
        assert manifest_path.exists()
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert raw["records"] == manifest.records

    def test_report_same_twice_from_fresh_preparer(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")
        config1 = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out1,
            dry_run=True,
        )
        config2 = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out2,
            dry_run=True,
        )
        _, r1 = LocalPreparer(config1).prepare(str(f))
        _, r2 = LocalPreparer(config2).prepare(str(f))
        assert r1.to_dict() == r2.to_dict()

    def test_report_written_to_disk(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=False,
        )
        preparer = LocalPreparer(config)
        preparer.prepare(str(f))
        report_path = tmp_path / "out" / "preparation_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["total_records"] >= 1

    def test_rejected_records_excluded_affects_count(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=True,
        )
        preparer = LocalPreparer(config)
        _, report = preparer.prepare(str(f))
        assert report.total_records >= 1
        assert report.rejected_records + report.accepted_records == report.total_records

    def test_contamination_exact_rejected(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        blocklist = tmp_path / "block.json"
        import json

        blocklist.write_text(json.dumps([REAL_TEXT]), encoding="utf-8")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            blocklist_path=str(blocklist),
        )
        _, report = LocalPreparer(config).prepare(str(f))
        assert "contamination:exact" in report.rejection_reasons

    def test_contamination_normalized_rejected(self, tmp_path):
        f = tmp_path / "input.txt"
        content = "  THE INDIAN EDUCATION SYSTEM  "
        f.write_text(content, encoding="utf-8")
        blocklist = tmp_path / "block.txt"
        blocklist.write_text("the indian education system", encoding="utf-8")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            blocklist_path=str(blocklist),
        )
        _, report = LocalPreparer(config).prepare(str(f))
        assert "contamination:normalized" in report.rejection_reasons

    def test_contamination_ngram_rejected(self, tmp_path):
        f = tmp_path / "input.txt"
        block_text = REAL_TEXT
        # Change one word so exact/normalized don't match but ngrams mostly overlap
        input_text = REAL_TEXT.replace("numeracy", "literacy")
        f.write_text(input_text, encoding="utf-8")
        blocklist = tmp_path / "block.json"
        import json

        blocklist.write_text(json.dumps([block_text]), encoding="utf-8")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            blocklist_path=str(blocklist),
        )
        _, report = LocalPreparer(config).prepare(str(f))
        reasons = report.rejection_reasons
        any_ngram = any(k.startswith("contamination:ngram") for k in reasons)
        assert any_ngram, f"No ngram key found in {reasons}"

    def test_clean_text_accepted_with_blocklist(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        blocklist = tmp_path / "block.txt"
        blocklist.write_text("completely unrelated content", encoding="utf-8")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            blocklist_path=str(blocklist),
        )
        _, report = LocalPreparer(config).prepare(str(f))
        assert "contamination:exact" not in report.rejection_reasons

    def test_contaminated_not_written_to_shards(self, tmp_path):
        import json

        f = tmp_path / "input.txt"
        f.write_text(REAL_TEXT, encoding="utf-8")
        blocklist = tmp_path / "block.json"
        blocklist.write_text(json.dumps([REAL_TEXT]), encoding="utf-8")
        out = str(tmp_path / "out")
        config = PreparationConfig(
            source_id="test",
            source_version="v1",
            license="cc-by-4.0",
            language="en",
            split="train",
            domain="web",
            output_dir=out,
            dry_run=False,
            blocklist_path=str(blocklist),
        )
        manifest, report = LocalPreparer(config).prepare(str(f))
        assert report.accepted_records == 0
        assert manifest.records == 0
        shard_dir = tmp_path / "out" / "shards"
        shard_files = list(shard_dir.glob("*.jsonl"))
        assert len(shard_files) == 0
