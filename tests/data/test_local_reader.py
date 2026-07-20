from __future__ import annotations

import pytest

from bharat.data.local_reader import read_local_text


class TestLocalReader:
    def test_txt_input(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("hello world", encoding="utf-8")
        records = read_local_text(str(f))
        assert len(records) == 1
        assert records[0].text == "hello world"
        assert records[0].source_path == str(f)
        assert records[0].line_number == 1

    def test_jsonl_input(self, tmp_path):
        f = tmp_path / "input.jsonl"
        f.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8")
        records = read_local_text(str(f))
        assert len(records) == 2
        assert records[0].text == "first"
        assert records[1].text == "second"

    def test_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Remote URLs"):
            read_local_text("https://example.com/data.txt")

    def test_invalid_jsonl_rejected(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSONL"):
            read_local_text(str(f))

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_local_text("/nonexistent/path.txt")

    def test_deterministic_ordering(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content", encoding="utf-8")
        r1 = read_local_text(str(f))
        r2 = read_local_text(str(f))
        assert r1 == r2
