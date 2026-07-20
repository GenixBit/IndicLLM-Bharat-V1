from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class PIISpan:
    start: int
    end: int
    pii_type: str
    text: str
    confidence: float


@dataclass(frozen=True)
class PIIConfig:
    enabled_types: tuple[str, ...] = (
        "email",
        "phone",
        "ip_address",
        "credit_card",
        "aadhaar",
        "pan",
        "url_credentials",
    )
    min_confidence: float = 0.5
    mask_char: str = "*"


class PIIDetector:
    _PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone": re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?:\s*(?:ext|x|ext.)\s*\d{1,5})?"
        ),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
        "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        "url_credentials": re.compile(r"\bhttps?://[^:\s]+:[^@\s]+@\S+\b"),
    }
    _LIKELY_PHONE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\d{7,15}")
    _LIKELY_CC_RE: ClassVar[re.Pattern[str]] = re.compile(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}")

    def __init__(self, config: PIIConfig | None = None) -> None:
        self.config = config or PIIConfig()

    def detect(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        if not text:
            return spans
        for pii_type in self.config.enabled_types:
            pattern = self._PATTERNS.get(pii_type)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                matched_text = match.group()
                confidence = self._compute_confidence(pii_type, matched_text)
                if confidence < self.config.min_confidence:
                    continue
                spans.append(
                    PIISpan(
                        start=match.start(),
                        end=match.end(),
                        pii_type=pii_type,
                        text=matched_text,
                        confidence=confidence,
                    )
                )
        spans.sort(key=lambda s: s.start)
        return spans

    def has_pii(self, text: str) -> bool:
        return len(self.detect(text)) > 0

    def redact(self, text: str) -> str:
        spans = self.detect(text)
        if not spans:
            return text
        result = list(text)
        for span in spans:
            mask = self.config.mask_char * (span.end - span.start)
            result[span.start : span.end] = list(mask)
        return "".join(result)

    def _compute_confidence(self, pii_type: str, matched: str) -> float:
        if pii_type == "email":
            return 0.95 if "@" in matched else 0.5
        if pii_type == "phone":
            digits = sum(1 for c in matched if c.isdigit())
            if 7 <= digits <= 15:
                return 0.9 if digits >= 10 else 0.6
            return 0.3
        if pii_type == "ip_address":
            parts = matched.split(".")
            if all(0 <= int(p) <= 255 for p in parts):
                return 0.95
            return 0.5
        if pii_type == "credit_card":
            if self._LIKELY_CC_RE.match(matched):
                return 0.85
            return 0.5
        if pii_type == "aadhaar":
            return 0.9
        if pii_type == "pan":
            return 0.95
        if pii_type == "url_credentials":
            return 0.95
        return 0.5
