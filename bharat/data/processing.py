"""Offline-safe data processing pipeline.

Composes Normalizer, LanguageIdentifier, QualityScorer, ExactDeduplicator,
FuzzyDeduplicator, PIIDetector, and SafetyFilter into a single deterministic
pipeline.  No data is downloaded; no remote URLs are read; no shards are
written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bharat.data.exact_dedup import ExactDedupConfig, ExactDeduplicator
from bharat.data.fuzzy_dedup import FuzzyDedupConfig, FuzzyDeduplicator
from bharat.data.language_id import LanguageIDConfig, LanguageIdentifier
from bharat.data.normalization import NormalizationConfig, Normalizer
from bharat.data.pii import PIIConfig, PIIDetector, PIISpan
from bharat.data.quality import QualityConfig, QualityScorer
from bharat.data.safety_filter import SafetyFilter, SafetyFilterConfig, SafetySpan


@dataclass(frozen=True)
class ProcessingConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    language_id: LanguageIDConfig = field(default_factory=LanguageIDConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    exact_dedup: ExactDedupConfig = field(default_factory=ExactDedupConfig)
    fuzzy_dedup: FuzzyDedupConfig = field(default_factory=FuzzyDedupConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    safety: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)


@dataclass(frozen=True)
class ProcessingDecision:
    accepted: bool
    normalized_text: str
    language: str
    quality_score: float
    reasons: tuple[str, ...]
    pii_spans: tuple[PIISpan, ...]
    safety_spans: tuple[SafetySpan, ...]


class DataProcessor:
    """Deterministic offline pipeline for processing raw text records."""

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()
        self.normalizer = Normalizer(self.config.normalization)
        self.lang_id = LanguageIdentifier(self.config.language_id)
        self.quality = QualityScorer(self.config.quality)
        self.exact_dedup = ExactDeduplicator(self.config.exact_dedup)
        self.fuzzy_dedup = FuzzyDeduplicator(self.config.fuzzy_dedup)
        self.pii = PIIDetector(self.config.pii)
        self.safety = SafetyFilter(self.config.safety)

    def process(self, text: str) -> ProcessingDecision:
        normalized = self.normalizer.normalize(text)
        lang_result = self.lang_id.identify(normalized or text)
        quality_decision = self.quality.evaluate(normalized or text)
        safety_result = self.safety.classify(normalized or text)
        pii_spans = self.pii.detect(normalized or text)
        reasons: list[str] = []
        if not safety_result.is_safe:
            reasons.append(f"unsafe:{','.join(safety_result.categories_violated)}")
        if pii_spans:
            reasons.append(f"pii:{','.join(s.pii_type for s in pii_spans)}")
        if not quality_decision.is_quality:
            reasons.extend(quality_decision.reasons)
        accepted = safety_result.is_safe and not pii_spans and quality_decision.is_quality
        exact_dup = False
        fuzzy_dup = False
        if accepted and normalized:
            exact_dup = not self.exact_dedup.add_document(normalized)
            fuzzy_dup = not self.fuzzy_dedup.add_document(normalized)
        if exact_dup:
            reasons.append("exact_duplicate")
        if fuzzy_dup:
            reasons.append("fuzzy_duplicate")
        reasons = sorted(set(reasons))
        accepted = accepted and not exact_dup and not fuzzy_dup
        return ProcessingDecision(
            accepted=accepted,
            normalized_text=normalized,
            language=lang_result.language,
            quality_score=quality_decision.score,
            reasons=tuple(reasons),
            pii_spans=tuple(pii_spans),
            safety_spans=tuple(safety_result.spans),
        )

    def process_batch(self, texts: list[str]) -> list[ProcessingDecision]:
        return [self.process(t) for t in texts]

    def reset_dedup(self) -> None:
        self.exact_dedup.reset()
        self.fuzzy_dedup.reset()
