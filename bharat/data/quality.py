from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class QualityScore:
    overall: float
    char_score: float
    word_score: float
    line_score: float
    features: dict[str, float]


@dataclass(frozen=True)
class QualityConfig:
    min_chars: int = 50
    max_chars: int = 1_000_000
    min_words: int = 10
    max_words: int = 200_000
    max_line_length: int = 1000
    min_alpha_ratio: float = 0.3
    max_repetition_ratio: float = 0.6
    max_bullet_ratio: float = 0.3
    max_ellipsis_ratio: float = 0.3
    max_punct_ratio: float = 0.3
    min_avg_word_len: float = 2.0
    max_avg_word_len: float = 20.0
    min_unique_word_ratio: float = 0.1


class QualityScorer:
    _URL_RE: ClassVar[re.Pattern[str]] = re.compile(r"https?://\S+")
    _BULLET_RE: ClassVar[re.Pattern[str]] = re.compile(r"^[\s]*[>\-]\s", re.MULTILINE)
    _ELLIPSIS_RE: ClassVar[re.Pattern[str]] = re.compile(r"\.{4,}")
    _REPEATED_WORD_RE: ClassVar[re.Pattern[str]] = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    _REPEATED_CHAR_RE: ClassVar[re.Pattern[str]] = re.compile(r"(.)\1{4,}")
    _NON_ALPHA_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9\s]")
    _PUNCT_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^\w\s]")
    _DIGIT_RE: ClassVar[re.Pattern[str]] = re.compile(r"\d")
    _UPPER_RE: ClassVar[re.Pattern[str]] = re.compile(r"[A-Z]")

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def score(self, text: str) -> QualityScore:
        if not text:
            return QualityScore(0.0, 0.0, 0.0, 0.0, {})
        features = self._extract_features(text)
        char_score = self._compute_char_score(features)
        word_score = self._compute_word_score(features)
        line_score = self._compute_line_score(features)
        overall = max(0.0, min(1.0, 0.4 * char_score + 0.35 * word_score + 0.25 * line_score))
        return QualityScore(
            overall=overall,
            char_score=char_score,
            word_score=word_score,
            line_score=line_score,
            features=features,
        )

    def is_quality(self, text: str, min_score: float = 0.5) -> bool:
        return self.score(text).overall >= min_score

    def _extract_features(self, text: str) -> dict[str, float]:
        chars = len(text)
        words_list = text.split()
        words = len(words_list)
        lines = text.splitlines()
        num_lines = len(lines)
        avg_word_len = statistics.mean(len(w) for w in words_list) if words_list else 0.0
        alpha_chars = sum(1 for c in text if c.isalpha())
        upper_chars = sum(1 for c in text if c.isupper())
        digit_chars = sum(1 for c in text if c.isdigit())
        punct_chars = len(self._PUNCT_RE.findall(text))
        space_chars = text.count(" ")
        bullet_lines = len(self._BULLET_RE.findall(text))
        ellipsis_count = len(self._ELLIPSIS_RE.findall(text))
        repeated_words = len(self._REPEATED_WORD_RE.findall(text))
        repeated_chars = len(self._REPEATED_CHAR_RE.findall(text))
        urls = len(self._URL_RE.findall(text))
        unique_words = len(set(w.lower() for w in words_list)) if words_list else 0
        line_lengths = [len(ln) for ln in lines if ln.strip()]
        return {
            "chars": float(chars),
            "words": float(words),
            "lines": float(num_lines),
            "avg_word_len": avg_word_len,
            "alpha_ratio": alpha_chars / max(chars, 1),
            "upper_ratio": upper_chars / max(alpha_chars, 1),
            "digit_ratio": digit_chars / max(chars, 1),
            "punct_ratio": punct_chars / max(chars, 1),
            "space_ratio": space_chars / max(chars, 1),
            "bullet_ratio": bullet_lines / max(num_lines, 1),
            "ellipsis_ratio": ellipsis_count / max(words, 1),
            "repeated_words": float(repeated_words),
            "repeated_char_ratio": repeated_chars / max(chars, 1),
            "url_count": float(urls),
            "unique_word_ratio": unique_words / max(words, 1),
            "avg_line_length": statistics.mean(line_lengths) if line_lengths else 0.0,
            "max_line_length": float(max(line_lengths)) if line_lengths else 0.0,
        }

    def _compute_char_score(self, f: dict[str, float]) -> float:
        score = 0.0
        if self.config.min_chars <= f["chars"] <= self.config.max_chars:
            score += 0.25
        if f["alpha_ratio"] >= self.config.min_alpha_ratio:
            score += 0.25
        if f["upper_ratio"] < 0.9:
            score += 0.1
        if f["digit_ratio"] < 0.5:
            score += 0.1
        if f["punct_ratio"] <= self.config.max_punct_ratio:
            score += 0.1
        if f["repeated_char_ratio"] < 0.01:
            score += 0.1
        for _ in range(int(f["repeated_words"])):
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _compute_word_score(self, f: dict[str, float]) -> float:
        score = 0.0
        if self.config.min_words <= f["words"] <= self.config.max_words:
            score += 0.3
        if self.config.min_avg_word_len <= f["avg_word_len"] <= self.config.max_avg_word_len:
            score += 0.2
        if f["unique_word_ratio"] >= self.config.min_unique_word_ratio:
            score += 0.3
        if f["ellipsis_ratio"] <= self.config.max_ellipsis_ratio:
            score += 0.1
        if f["url_count"] <= 5:
            score += 0.1
        return max(0.0, min(1.0, score))

    def _compute_line_score(self, f: dict[str, float]) -> float:
        score = 0.0
        if f["lines"] >= 2:
            score += 0.3
        if f["avg_line_length"] > 0:
            score += 0.2
        if f["max_line_length"] <= self.config.max_line_length:
            score += 0.2
        if f["bullet_ratio"] <= self.config.max_bullet_ratio:
            score += 0.3
        return max(0.0, min(1.0, score))
