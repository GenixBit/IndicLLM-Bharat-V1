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
class QualityDecision:
    is_quality: bool
    score: float
    reasons: tuple[str, ...]
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


_REASON_CODES: dict[str, str] = {
    "min_chars": "too_short",
    "max_chars": "too_long",
    "min_words": "too_few_words",
    "max_words": "too_many_words",
    "min_alpha_ratio": "low_alpha_ratio",
    "max_punct_ratio": "too_many_punctuation",
    "max_ellipsis_ratio": "too_many_ellipses",
    "max_bullet_ratio": "too_many_bullets",
    "min_avg_word_len": "avg_word_too_short",
    "max_avg_word_len": "avg_word_too_long",
    "min_unique_word_ratio": "too_few_unique_words",
    "max_line_length": "line_too_long",
    "repeated_words": "excessive_repetition",
    "url_count": "too_many_urls",
    "repeated_char": "excessive_char_repetition",
    "lines": "too_few_lines",
}


class QualityScorer:
    _URL_RE: ClassVar[re.Pattern[str]] = re.compile(r"https?://\S+")
    _BULLET_RE: ClassVar[re.Pattern[str]] = re.compile(r"^[\s]*[>\-]\s", re.MULTILINE)
    _ELLIPSIS_RE: ClassVar[re.Pattern[str]] = re.compile(r"\.{4,}")
    _REPEATED_WORD_RE: ClassVar[re.Pattern[str]] = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    _REPEATED_CHAR_RE: ClassVar[re.Pattern[str]] = re.compile(r"(.)\1{4,}")
    _PUNCT_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^\w\s]")

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.min_chars <= 0:
            raise ValueError("min_chars must be > 0")
        if self.config.max_chars < self.config.min_chars:
            raise ValueError("max_chars must be >= min_chars")
        if self.config.min_words <= 0:
            raise ValueError("min_words must be > 0")
        if self.config.max_words < self.config.min_words:
            raise ValueError("max_words must be >= min_words")
        if not 0.0 <= self.config.min_alpha_ratio <= 1.0:
            raise ValueError("min_alpha_ratio must be in [0.0, 1.0]")
        if not 0.0 <= self.config.max_punct_ratio <= 1.0:
            raise ValueError("max_punct_ratio must be in [0.0, 1.0]")
        if not 0.0 <= self.config.max_ellipsis_ratio <= 1.0:
            raise ValueError("max_ellipsis_ratio must be in [0.0, 1.0]")
        if not 0.0 <= self.config.max_bullet_ratio <= 1.0:
            raise ValueError("max_bullet_ratio must be in [0.0, 1.0]")

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

    def evaluate(self, text: str) -> QualityDecision:
        qs = self.score(text)
        reasons: list[str] = []
        f = qs.features
        if not f:
            return QualityDecision(is_quality=False, score=0.0, reasons=("empty",), features={})
        if f["chars"] < self.config.min_chars:
            reasons.append(_REASON_CODES["min_chars"])
        if f["chars"] > self.config.max_chars:
            reasons.append(_REASON_CODES["max_chars"])
        if f["words"] < self.config.min_words:
            reasons.append(_REASON_CODES["min_words"])
        if f["words"] > self.config.max_words:
            reasons.append(_REASON_CODES["max_words"])
        if f["alpha_ratio"] < self.config.min_alpha_ratio:
            reasons.append(_REASON_CODES["min_alpha_ratio"])
        if f["punct_ratio"] > self.config.max_punct_ratio:
            reasons.append(_REASON_CODES["max_punct_ratio"])
        if f["ellipsis_ratio"] > self.config.max_ellipsis_ratio:
            reasons.append(_REASON_CODES["max_ellipsis_ratio"])
        if f["bullet_ratio"] > self.config.max_bullet_ratio:
            reasons.append(_REASON_CODES["max_bullet_ratio"])
        if "avg_word_len" in f:
            if f["avg_word_len"] < self.config.min_avg_word_len:
                reasons.append(_REASON_CODES["min_avg_word_len"])
            if f["avg_word_len"] > self.config.max_avg_word_len:
                reasons.append(_REASON_CODES["max_avg_word_len"])
        if f["unique_word_ratio"] < self.config.min_unique_word_ratio:
            reasons.append(_REASON_CODES["min_unique_word_ratio"])
        if f["max_line_length"] > self.config.max_line_length:
            reasons.append(_REASON_CODES["max_line_length"])
        if int(f["repeated_words"]) > 5:
            reasons.append(_REASON_CODES["repeated_words"])
        if f["url_count"] > 10:
            reasons.append(_REASON_CODES["url_count"])
        if f["repeated_char_ratio"] > 0.05:
            reasons.append(_REASON_CODES["repeated_char"])
        if f["lines"] < 2:
            reasons.append(_REASON_CODES["lines"])
        is_quality = qs.overall >= 0.5 and not reasons
        return QualityDecision(
            is_quality=is_quality, score=qs.overall, reasons=tuple(reasons), features=f
        )

    def is_quality(self, text: str, min_score: float = 0.5) -> bool:
        return self.evaluate(text).is_quality

    def _extract_features(self, text: str) -> dict[str, float]:
        chars = len(text)
        words_list = text.split()
        words = len(words_list)
        lines = text.splitlines()
        num_lines = len(lines)
        avg_word_len = statistics.mean(len(w) for w in words_list) if words_list else 0.0
        alpha_chars = sum(1 for c in text if c.isalpha())
        punct_chars = len(self._PUNCT_RE.findall(text))
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
            "punct_ratio": punct_chars / max(chars, 1),
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
            score += 0.3
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
