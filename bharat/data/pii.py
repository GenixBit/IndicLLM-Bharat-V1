from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

_PII_PRIORITY: dict[str, int] = {
    "url_credentials": 1,
    "email": 2,
    "credit_card": 3,
    "aadhaar": 4,
    "pan": 5,
    "ip_address": 6,
    "phone": 7,
}


def _luhn_checksum(digits: str) -> bool:
    total = 0
    alternate = False
    for d in reversed(digits):
        n = ord(d) - 48
        if alternate:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alternate = not alternate
    return total % 10 == 0


def _resolve_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s.start, -_PII_PRIORITY.get(s.pii_type, 99)))
    merged: list[PIISpan] = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        last = merged[-1]
        if span.start >= last.end:
            merged.append(span)
            continue
        last_pri = _PII_PRIORITY.get(last.pii_type, 99)
        span_pri = _PII_PRIORITY.get(span.pii_type, 99)
        if span_pri < last_pri:
            merged[-1] = span
        elif span_pri == last_pri and span.end > last.end:
            merged[-1] = PIISpan(
                start=last.start,
                end=max(last.end, span.end),
                pii_type=last.pii_type,
                text=last.text,
                confidence=max(last.confidence, span.confidence),
            )
    return merged


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
                if pii_type == "credit_card" and not _luhn_checksum(
                    "".join(c for c in matched_text if c.isdigit())
                ):
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
        return _resolve_overlaps(spans)

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
            if len(parts) != 4:
                return 0.3
            if all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
                return 0.95
            return 0.3
        if pii_type == "credit_card":
            return 0.85
        if pii_type == "aadhaar":
            return 0.7
        if pii_type == "pan":
            return 0.95
        if pii_type == "url_credentials":
            return 0.95
        return 0.5
