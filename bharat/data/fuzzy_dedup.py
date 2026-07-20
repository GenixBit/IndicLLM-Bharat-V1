from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from bharat.data.normalization import NormalizationConfig, Normalizer


@dataclass(frozen=True)
class FuzzyDedupConfig:
    n_gram_size: int = 5
    num_permutations: int = 128
    threshold: float = 0.8
    normalize: bool = True
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)


_NORMALIZER = Normalizer()
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s]")


def _default_minhash_signature(text: str, n_gram_size: int, num_perm: int) -> list[int]:
    words = _NON_ALNUM_RE.sub("", text).lower().split()
    if len(words) < n_gram_size:
        n_gram_size = max(1, len(words))
    n_grams: set[int] = set()
    for i in range(len(words) - n_gram_size + 1):
        ng = " ".join(words[i : i + n_gram_size])
        h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
        n_grams.add(h)
    if not n_grams:
        return [0] * num_perm
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
        self._signatures: list[tuple[str, list[int]]] = []

    def add_document(self, text: str) -> bool:
        if not text:
            return False
        sig = self._compute_signature(text)
        for _, existing in self._signatures:
            if _jaccard_similarity(sig, existing) >= self.config.threshold:
                return False
        self._signatures.append((text, sig))
        return True

    def is_duplicate(self, text: str) -> bool:
        if not text:
            return False
        sig = self._compute_signature(text)
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

    def _compute_signature(self, text: str) -> list[int]:
        if self.config.normalize:
            text = _NORMALIZER.normalize(text)
        return _default_minhash_signature(
            text, self.config.n_gram_size, self.config.num_permutations
        )
