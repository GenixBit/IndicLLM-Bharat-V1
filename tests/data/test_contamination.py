from __future__ import annotations

import json

import pytest

from bharat.data.contamination import ContaminationChecker


class TestContaminationChecker:
    def test_exact_contaminated(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("contaminated text here\nsafe reference text\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_exact("contaminated text here")
        assert result.is_contaminated
        assert result.method == "exact"

    def test_clean_text_not_contaminated(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("blocked text\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_exact("completely clean text")
        assert not result.is_contaminated

    def test_normalized_match(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("Contaminated   TEXT\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_normalized("contaminated text")
        assert result.is_contaminated
        assert result.method == "normalized"

    def test_ngram_near_overlap(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("the quick brown fox jumps over the lazy dog\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_ngram("the quick brown fox leaps over the lazy dog", n=3)
        assert result.is_contaminated
        assert "ngram" in result.method

    def test_safe_text_not_contaminated_by_ngram(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("the quick brown fox jumps over the lazy dog\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_ngram("completely unrelated text about something else entirely", n=3)
        assert not result.is_contaminated

    def test_empty_blocklist(self):
        checker = ContaminationChecker()
        result = checker.check_exact("any text")
        assert not result.is_contaminated

    def test_deterministic(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("test content\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        r1 = checker.check_exact("test content")
        r2 = checker.check_exact("test content")
        assert r1 == r2

    def test_check_all_returns_exact_match(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("exact match text\nalmost the same text now with changes\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        result = checker.check_all("exact match text")
        assert result.is_contaminated
        assert result.method == "exact"

    def test_reset_clears_blocklist(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("something\n")
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        assert checker.check_exact("something").is_contaminated
        checker.reset()
        assert not checker.check_exact("something").is_contaminated

    def test_json_blocklist(self, tmp_path):
        blocklist = tmp_path / "blocklist.json"
        blocklist.write_text(json.dumps(["text one", "text two"]))
        checker = ContaminationChecker()
        checker.load_blocklist(str(blocklist))
        assert checker.check_exact("text one").is_contaminated
        assert checker.check_exact("not present").is_contaminated is False

    def test_missing_file_raises(self):
        checker = ContaminationChecker()
        with pytest.raises(FileNotFoundError, match="Blocklist not found"):
            checker.load_blocklist("/nonexistent/path.txt")
