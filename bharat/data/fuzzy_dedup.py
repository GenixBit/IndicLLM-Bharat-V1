from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from bharat.data.normalization import NormalizationConfig, Normalizer


@dataclass(frozen=True)
class FuzzyDedupConfig:
    n_gram_size: int = 5
    num_permutations: int = 128
    threshold: float = 0.8
    normalize: bool = True
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)


_GLOBAL_NORMALIZER = Normalizer()


def _unicode_words(text: str) -> list[str]:
    if not text:
        return []
    if not any(c.isalnum() for c in text):
        return []
    return text.split()


def _default_minhash_signature(text: str, n_gram_size: int, num_perm: int) -> list[int] | None:
    words = _unicode_words(text)
    n_grams: set[int] = set()
    if not words:
        return None
    if len(words) >= n_gram_size:
        for i in range(len(words) - n_gram_size + 1):
            ng = " ".join(words[i : i + n_gram_size])
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
            n_grams.add(h)
    else:
        effective = max(1, min(n_gram_size, len(text)))
        for i in range(len(text) - effective + 1):
            ng = text[i : i + effective]
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
            n_grams.add(h)
    if not n_grams:
        return None
    a_coeffs = [2 * i + 1 for i in range(num_perm)]
    b_coeffs = [3 * i + 7 for i in range(num_perm)]
    max_hash = 2**32 - 1
    signature: list[int] = []
    for a, b in zip(a_coeffs, b_coeffs, strict=False):
        min_val = max_hash
        for h_val in n_grams:
            perm = (a * h_val + b) % max_hash
            if perm < min_val:
                min_val = perm
        signature.append(min_val)
    return signature


def _jaccard_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    return matches / len(sig_a)


class FuzzyDeduplicator:
    def __init__(self, config: FuzzyDedupConfig | None = None) -> None:
        self.config = config or FuzzyDedupConfig()
        if self.config.n_gram_size < 1:
            raise ValueError("n_gram_size must be >= 1")
        if self.config.num_permutations < 1:
            raise ValueError("num_permutations must be >= 1")
        if not 0.0 <= self.config.threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self._signatures: list[tuple[str, list[int]]] = []

    def add_document(self, text: str) -> bool:
        if not text:
            return False
        sig = self._compute_signature(text)
        if sig is None:
            return False
        for _, existing in self._signatures:
            if _jaccard_similarity(sig, existing) >= self.config.threshold:
                return False
        self._signatures.append((text, sig))
        return True

    def is_duplicate(self, text: str) -> bool:
        if not text:
            return False
        sig = self._compute_signature(text)
        if sig is None:
            return False
        for _, existing in self._signatures:
            if _jaccard_similarity(sig, existing) >= self.config.threshold:
                return True
        return False

    def filter(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        for text in texts:
            if self.add_document(text):
                result.append(text)
        return result

    def reset(self) -> None:
        self._signatures.clear()

    @property
    def seen_count(self) -> int:
        return len(self._signatures)

    def _compute_signature(self, text: str) -> list[int] | None:
        if self.config.normalize:
            if self.config.normalization_config != FuzzyDedupConfig().normalization_config:
                normalizer = Normalizer(self.config.normalization_config)
            else:
                normalizer = _GLOBAL_NORMALIZER
            text = normalizer.normalize(text)
        return _default_minhash_signature(
            text, self.config.n_gram_size, self.config.num_permutations
        )
