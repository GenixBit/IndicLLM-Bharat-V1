from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: str = "NFC"
    collapse_whitespace: bool = True
    strip: bool = True
    remove_control: bool = True
    remove_zero_width: bool = True
    normalize_line_endings: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    lowercase: bool = False
    max_length: int = 0


@dataclass(frozen=True)
class NormalizationResult:
    original: str
    normalized: str
    config: NormalizationConfig


class Normalizer:
    _URL_RE: ClassVar[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+")
    _EMAIL_RE: ClassVar[re.Pattern[str]] = re.compile(r"\S+@\S+\.\S+")
    _LINE_ENDING_RE: ClassVar[re.Pattern[str]] = re.compile(r"\r\n|\r(?!\n)")
    _WHITESPACE_RE: ClassVar[re.Pattern[str]] = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\n{3,}")
    _CONTROL_CHARS_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    )
    _ZERO_WIDTH_RE: ClassVar[re.Pattern[str]] = re.compile(
        "[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufeff]"
    )

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> str:
        if not text:
            return text
        result = text
        if self.config.normalize_line_endings:
            result = self._LINE_ENDING_RE.sub("\n", result)
        if self.config.remove_control:
            result = self._CONTROL_CHARS_RE.sub("", result)
        if self.config.remove_zero_width:
            result = self._ZERO_WIDTH_RE.sub("", result)
        if self.config.unicode_form:
            result = unicodedata.normalize(self.config.unicode_form, result)  # type: ignore[arg-type]
        if self.config.remove_urls:
            result = self._URL_RE.sub("", result)
        if self.config.remove_emails:
            result = self._EMAIL_RE.sub("", result)
        if self.config.collapse_whitespace:
            result = self._WHITESPACE_RE.sub(" ", result)
            result = self._MULTI_NEWLINE_RE.sub("\n\n", result)
        if self.config.strip:
            result = result.strip()
        if self.config.lowercase:
            result = result.lower()
        if self.config.max_length > 0 and len(result) > self.config.max_length:
            result = result[: self.config.max_length]
        return result

    def normalize_with_meta(self, text: str) -> NormalizationResult:
        return NormalizationResult(
            original=text, normalized=self.normalize(text), config=self.config
        )
