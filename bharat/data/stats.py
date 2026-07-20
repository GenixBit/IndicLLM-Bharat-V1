from __future__ import annotations

from dataclasses import dataclass, field

from bharat.data.processing import DataProcessor


@dataclass(frozen=True)
class DatasetStatistics:
    record_count: int = 0
    total_chars: int = 0
    total_utf8_bytes: int = 0
    avg_chars: float = 0.0
    avg_words: float = 0.0
    language_distribution: dict[str, int] = field(default_factory=dict)
    quality_score_distribution: dict[str, int] = field(default_factory=dict)
    pii_rejection_count: int = 0
    safety_rejection_count: int = 0
    duplicate_rejection_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0


def _score_bucket(score: float) -> str:
    if score >= 0.9:
        return "0.9-1.0"
    elif score >= 0.7:
        return "0.7-0.9"
    elif score >= 0.5:
        return "0.5-0.7"
    elif score >= 0.3:
        return "0.3-0.5"
    elif score >= 0.1:
        return "0.1-0.3"
    return "0.0-0.1"


def compute_statistics(
    texts: list[str],
    processor: DataProcessor | None = None,
) -> DatasetStatistics:
    if processor is None:
        processor = DataProcessor()
    if not texts:
        return DatasetStatistics()

    decisions = processor.process_batch(texts)
    total_chars = sum(len(t) for t in texts)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    lang_dist: dict[str, int] = {}
    quality_dist: dict[str, int] = {}
    pii_rej = 0
    safety_rej = 0
    dup_rej = 0
    accepted = 0
    rejected = 0

    for d in decisions:
        lang_dist[d.language] = lang_dist.get(d.language, 0) + 1
        bucket = _score_bucket(d.quality_score)
        quality_dist[bucket] = quality_dist.get(bucket, 0) + 1
        if d.accepted:
            accepted += 1
        else:
            rejected += 1
        for reason in d.reasons:
            if reason.startswith("pii:"):
                pii_rej += 1
            elif "unsafe" in reason:
                safety_rej += 1
            elif "duplicate" in reason:
                dup_rej += 1

    return DatasetStatistics(
        record_count=len(texts),
        total_chars=total_chars,
        total_utf8_bytes=total_bytes,
        avg_chars=total_chars / len(texts) if texts else 0.0,
        avg_words=total_words / len(texts) if texts else 0.0,
        language_distribution=lang_dist,
        quality_score_distribution=quality_dist,
        pii_rejection_count=pii_rej,
        safety_rejection_count=safety_rej,
        duplicate_rejection_count=dup_rej,
        accepted_count=accepted,
        rejected_count=rejected,
    )
