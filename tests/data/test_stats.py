from __future__ import annotations

from bharat.data.processing import DataProcessor
from bharat.data.stats import DatasetStatistics, compute_statistics


class TestComputeStatistics:
    def test_empty_input(self):
        stats = compute_statistics([])
        assert stats.record_count == 0
        assert stats.total_chars == 0

    def test_single_record(self):
        stats = compute_statistics(["hello world"])
        assert stats.record_count == 1
        assert stats.total_chars == 11

    def test_basic_aggregation(self):
        texts = [
            "This is a test document with enough text for quality scoring.\nIt has multiple lines too.",
            "Another document that should pass quality checks as well.\nSecond line here.",
        ]
        stats = compute_statistics(texts)
        assert stats.record_count == 2
        assert stats.total_chars > 0
        assert stats.avg_chars > 0
        assert stats.avg_words > 0

    def test_language_distribution(self):
        texts = [
            "English text document for testing purposes.\nThis has multiple lines for quality.",
        ]
        stats = compute_statistics(texts)
        assert stats.language_distribution is not None
        assert sum(stats.language_distribution.values()) == 1

    def test_rejected_counts(self):
        stats = compute_statistics([""])
        assert stats.rejected_count == 1
        assert stats.accepted_count == 0

    def test_deterministic(self):
        texts = [
            "First test document with proper content.\nIt has lines.",
            "Second document for testing.\nAnother line here.",
        ]
        s1 = compute_statistics(texts)
        s2 = compute_statistics(texts)
        assert s1.record_count == s2.record_count
        assert s1.accepted_count == s2.accepted_count
        assert s1.rejected_count == s2.rejected_count

    def test_custom_processor(self):
        processor = DataProcessor()
        texts = ["custom processor test.\nSecond line."]
        stats = compute_statistics(texts, processor=processor)
        assert isinstance(stats, DatasetStatistics)

    def test_non_empty_stats_object(self):
        texts = [
            "Quality document first line.\nSecond line for testing purposes.",
            "Another quality document.\nSecond line content here.",
        ]
        stats = compute_statistics(texts)
        assert isinstance(stats.quality_score_distribution, dict)
        assert len(stats.quality_score_distribution) >= 1
