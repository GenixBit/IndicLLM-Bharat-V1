from __future__ import annotations

from bharat.tokenizer.evaluate import (
    all_metrics,
    code_efficiency,
    compression_ratio,
    fertility,
    language_wise_fertility,
    top_k_common_tokens,
    top_k_rare_tokens,
)

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a programming language.",
    "Machine learning is transforming artificial intelligence.",
    "Natural language processing enables computers to understand text.",
]

SAMPLE_CODE = [
    'def hello():\n    print("Hello, world!")',
    "import pandas as pd\nimport numpy as np",
]


class TestCompressionRatio:
    def test_positive_value(self, tiny_tokenizer) -> None:
        cr = compression_ratio(tiny_tokenizer, SAMPLE_TEXTS)
        assert cr > 0

    def test_reasonable_ratio(self, tiny_tokenizer) -> None:
        cr = compression_ratio(tiny_tokenizer, SAMPLE_TEXTS)
        assert 1.0 < cr < 20.0


class TestFertility:
    def test_positive_value(self, tiny_tokenizer) -> None:
        f = fertility(tiny_tokenizer, SAMPLE_TEXTS)
        assert f > 0

    def test_reasonable_value(self, tiny_tokenizer) -> None:
        f = fertility(tiny_tokenizer, SAMPLE_TEXTS)
        assert 0.5 < f < 5.0


class TestTopKTokens:
    def test_common_tokens_count(self, tiny_tokenizer) -> None:
        common = top_k_common_tokens(tiny_tokenizer, SAMPLE_TEXTS, k=5)
        assert len(common) <= 5

    def test_common_tokens_format(self, tiny_tokenizer) -> None:
        common = top_k_common_tokens(tiny_tokenizer, SAMPLE_TEXTS, k=3)
        for token, count in common:
            assert isinstance(token, str)
            assert isinstance(count, int)
            assert count > 0

    def test_rare_tokens_count(self, tiny_tokenizer) -> None:
        rare = top_k_rare_tokens(tiny_tokenizer, SAMPLE_TEXTS, k=5)
        assert len(rare) <= 5


class TestLanguageWiseFertility:
    def test_returns_dict(self, tiny_tokenizer) -> None:
        texts_by_lang = {
            "en": SAMPLE_TEXTS,
            "code": SAMPLE_CODE,
        }
        result = language_wise_fertility(tiny_tokenizer, texts_by_lang)
        assert isinstance(result, dict)
        assert "en" in result
        assert "code" in result

    def test_positive_values(self, tiny_tokenizer) -> None:
        texts_by_lang = {"en": SAMPLE_TEXTS}
        result = language_wise_fertility(tiny_tokenizer, texts_by_lang)
        assert result["en"] > 0


class TestCodeEfficiency:
    def test_positive_value(self, tiny_tokenizer) -> None:
        ce = code_efficiency(tiny_tokenizer, SAMPLE_CODE)
        assert ce > 0


class TestAllMetrics:
    def test_returns_all_metrics(self, tiny_tokenizer) -> None:
        metrics = all_metrics(tiny_tokenizer, SAMPLE_TEXTS)
        expected = {"compression_ratio", "fertility", "code_efficiency"}
        assert set(metrics.keys()) == expected

    def test_positive_values(self, tiny_tokenizer) -> None:
        metrics = all_metrics(tiny_tokenizer, SAMPLE_TEXTS)
        for value in metrics.values():
            assert value > 0
