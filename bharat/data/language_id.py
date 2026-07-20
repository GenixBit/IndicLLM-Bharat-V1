from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_SCRIPT_INDIC_MAP: dict[str, str] = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "ORIYA": "or",
    "TAMIL": "ta",
    "TELUGU": "te",
    "KANNADA": "kn",
    "MALAYALAM": "ml",
    "SINHALA": "si",
    "THAI": "th",
    "LAO": "lo",
    "KHMER": "km",
    "MYANMAR": "my",
    "ARABIC": "ar",
    "HEBREW": "he",
    "CYRILLIC": "ru",
    "GREEK": "el",
}


@dataclass(frozen=True)
class LanguageIDConfig:
    script_fallback: bool = True
    min_text_length: int = 20
    confidence_threshold: float = 0.0
    model_path: str = ""


@dataclass(frozen=True)
class LanguageIDResult:
    language: str
    confidence: float
    script: str
    method: str


class LanguageIdentifier:
    def __init__(self, config: LanguageIDConfig | None = None) -> None:
        self.config = config or LanguageIDConfig()

    def identify(self, text: str) -> LanguageIDResult:
        script = self._detect_script(text) if text else "unknown"
        if len(text) < self.config.min_text_length:
            if self.config.script_fallback and script != "Latin" and script != "unknown":
                lang = _SCRIPT_INDIC_MAP.get(script, "unknown")
                return LanguageIDResult(
                    language=lang,
                    confidence=0.8 if lang != "unknown" else 0.0,
                    script=script,
                    method="script_fallback",
                )
            return LanguageIDResult(
                language="unknown", confidence=0.0, script=script, method="too_short"
            )
        result = self._try_langdetect(text)
        if result is not None:
            if result.confidence >= self.config.confidence_threshold:
                return result
            if not self.config.script_fallback:
                return LanguageIDResult(
                    language="unknown",
                    confidence=0.0,
                    script=self._detect_script(text),
                    method="unknown",
                )
        if self.config.script_fallback:
            return self._identify_by_script(text)
        return LanguageIDResult(
            language="unknown", confidence=0.0, script=self._detect_script(text), method="none"
        )

    def identify_batch(self, texts: list[str]) -> list[LanguageIDResult]:
        return [self.identify(t) for t in texts]

    def _try_langdetect(self, text: str) -> LanguageIDResult | None:
        try:
            import langdetect as _ld

            try:
                probs = _ld.detect_langs(text)
            except Exception:
                probs = None
            if probs:
                best = probs[0]
                prob = getattr(best, "prob", 1.0)
                lang = str(getattr(best, "lang", ""))
                if lang:
                    return LanguageIDResult(
                        language=lang,
                        confidence=min(1.0, prob),
                        script=self._detect_script(text),
                        method="langdetect",
                    )
        except ImportError:
            pass
        except Exception:
            pass
        return None

    def _identify_by_script(self, text: str) -> LanguageIDResult:
        script = self._detect_script(text)
        lang = _SCRIPT_INDIC_MAP.get(script, "unknown")
        confidence = 0.8 if lang != "unknown" else 0.0
        return LanguageIDResult(
            language=lang, confidence=confidence, script=script, method="script_fallback"
        )

    @staticmethod
    def _detect_script(text: str) -> str:
        script_counts: dict[str, int] = {}
        for ch in text:
            if ch.isspace() or ch.isascii():
                continue
            try:
                script = unicodedata.name(ch, "").split(" ")[0]
                if script:
                    script_counts[script] = script_counts.get(script, 0) + 1
            except ValueError:
                continue
        if not script_counts:
            return "Latin"
        dominant = max(script_counts, key=lambda k: script_counts[k])
        return dominant
