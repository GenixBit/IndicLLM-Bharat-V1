"""Heuristic pre-filter for potentially unsafe content.

This module uses simple pattern matching to flag content that may
contain hate speech, profanity, violence, sexual content, or spam.

**Important**: This is a heuristic pre-filter only, not a legal or
safety guarantee. It may produce false positives and false negatives.
Pattern-based filtering cannot replace human review or dedicated
content moderation systems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SafetySpan:
    start: int
    end: int
    category: str
    text: str
    confidence: float


@dataclass(frozen=True)
class SafetyResult:
    is_safe: bool
    categories_violated: tuple[str, ...]
    spans: tuple[SafetySpan, ...]
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SafetyFilterConfig:
    enabled_categories: tuple[str, ...] = (
        "hate_speech",
        "profanity",
        "violence",
        "sexual",
        "spam",
    )
    min_confidence: float = 0.3
    threshold: float = 0.5
    blocklist_path: str = ""


class SafetyFilter:
    """Heuristic pre-filter only — not a legal or safety guarantee."""

    _TOXIC_PATTERNS: ClassVar[dict[str, list[re.Pattern[str]]]] = {
        "hate_speech": [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in [
                r"\b(?:fool|idiot|stupid|dumb|moron|retard)\b",
            ]
        ],
        "profanity": [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in [
                r"\b(?:damn|shit|fuck|ass|crap|bastard|bitch)\b",
            ]
        ],
        "violence": [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in [
                r"\b(?:kill|murder|slaughter|massacre|torture|bomb|explosion|deadly|attack)\b",
            ]
        ],
        "sexual": [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in [
                r"\b(?:porn|xxx|sex|nsfw|onlyfans)\b",
            ]
        ],
        "spam": [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in [
                r"\b(?:click here|subscribe|limited time|act now|free money|congratulations.*won)\b",
            ]
        ],
    }

    def __init__(self, config: SafetyFilterConfig | None = None) -> None:
        self.config = config or SafetyFilterConfig()
        if not 0.0 <= self.config.threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.config.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0.0, 1.0]")

    def classify(self, text: str) -> SafetyResult:
        if not text:
            return SafetyResult(
                is_safe=True, categories_violated=(), spans=(), score=1.0, reasons=()
            )
        spans: list[SafetySpan] = []
        categories_hit: set[str] = set()
        total_confidence = 0.0
        for category in self.config.enabled_categories:
            patterns = self._TOXIC_PATTERNS.get(category, [])
            cat_confidence = 0.0
            for pattern in patterns:
                for match in pattern.finditer(text):
                    confidence = min(1.0, 0.3 + 0.1 * len(match.group().split()))
                    if confidence >= self.config.min_confidence:
                        spans.append(
                            SafetySpan(
                                start=match.start(),
                                end=match.end(),
                                category=category,
                                text=match.group(),
                                confidence=confidence,
                            )
                        )
                        cat_confidence = max(cat_confidence, confidence)
            if cat_confidence > 0:
                categories_hit.add(category)
                total_confidence += cat_confidence
        score = max(0.0, 1.0 - total_confidence / len(categories_hit)) if categories_hit else 1.0
        is_safe = score >= self.config.threshold
        spans.sort(key=lambda s: s.start)
        reasons = tuple(f"{s.category}:{s.text}" for s in spans if s.category in categories_hit)
        return SafetyResult(
            is_safe=is_safe,
            categories_violated=tuple(sorted(categories_hit)),
            spans=tuple(spans),
            score=score,
            reasons=reasons,
        )

    def is_safe(self, text: str) -> bool:
        return self.classify(text).is_safe

    def filter(self, texts: list[str]) -> list[str]:
        return [t for t in texts if self.is_safe(t)]
