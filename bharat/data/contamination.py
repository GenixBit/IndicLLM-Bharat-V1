from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContaminationConfig:
    ngram_threshold: float = 0.5


@dataclass(frozen=True)
class ContaminationResult:
    is_contaminated: bool
    method: str
    score: float
    matched_sources: tuple[str, ...] = ()


_CONTAMINATION_RNG = 1315423911


def _hash_text(text: str) -> int:
    h = _CONTAMINATION_RNG
    for b in text.encode("utf-8"):
        h ^= (h << 5) + b + (h >> 2)
    return h & 0x7FFFFFFF


class ContaminationChecker:
    def __init__(self, config: ContaminationConfig | None = None) -> None:
        self._config = config or ContaminationConfig()
        self._entries: list[str] = []
        self._entry_set: set[str] = set()
        self._exact_hashes: set[str] = set()
        self._normalized_set: set[str] = set()
        self._ngram_cache: dict[int, list[set[int]]] = {}

    def load_blocklist(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Blocklist not found: {path}")
        if path.suffix == ".json":
            import json

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                entries: list[str] = data
            elif isinstance(data, dict) and "texts" in data:
                entries = data["texts"]
            else:
                raise ValueError(f"Unsupported JSON blocklist format in {path}")
        else:
            with path.open("r", encoding="utf-8") as f:
                entries = [line.strip() for line in f if line.strip()]
        for entry in entries:
            if entry not in self._entry_set:
                self._entries.append(entry)
                self._entry_set.add(entry)
                self._exact_hashes.add(self._sha256(entry))
                self._normalized_set.add(self._normalize(entry))
                self._ngram_cache.clear()

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _text_ngrams(self, text: str, n: int) -> set[int]:
        words = text.split()
        if len(words) < n:
            effective = min(n, len(text))
            if effective < 1:
                return set()
            return {
                _hash_text(text[i : i + effective])
                for i in range(max(1, len(text) - effective + 1))
            }
        ngrams: set[int] = set()
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i : i + n])
            ngrams.add(_hash_text(ng))
        return ngrams

    def _get_ngrams_for(self, n: int) -> list[set[int]]:
        if n not in self._ngram_cache:
            self._ngram_cache[n] = [
                self._text_ngrams(e, n) for e in self._entries
            ]
        return self._ngram_cache[n]

    def check_exact(self, text: str) -> ContaminationResult:
        if not text or not self._entries:
            return ContaminationResult(False, "exact", 0.0)
        h = self._sha256(text)
        if h in self._exact_hashes:
            return ContaminationResult(True, "exact", 1.0)
        return ContaminationResult(False, "exact", 0.0)

    def check_normalized(self, text: str) -> ContaminationResult:
        if not text or not self._entries:
            return ContaminationResult(False, "normalized", 0.0)
        normalized = self._normalize(text)
        if normalized in self._normalized_set:
            return ContaminationResult(True, "normalized", 1.0)
        return ContaminationResult(False, "normalized", 0.0)

    def check_ngram(self, text: str, n: int = 5) -> ContaminationResult:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not text or not self._entries:
            return ContaminationResult(False, f"ngram_{n}", 0.0)
        text = self._normalize(text)
        text_ngs = self._text_ngrams(text, n)
        if not text_ngs:
            return ContaminationResult(False, f"ngram_{n}", 0.0)
        block_ngram_list = self._get_ngrams_for(n)
        max_overlap = 0.0
        matched: list[str] = []
        for i, block_ngs in enumerate(block_ngram_list):
            if not block_ngs:
                continue
            common = len(text_ngs & block_ngs)
            overlap = common / max(len(text_ngs), len(block_ngs))
            if overlap > max_overlap:
                max_overlap = overlap
            if overlap >= self._config.ngram_threshold:
                matched.append(self._sha256(self._entries[i])[:12])
        is_contaminated = max_overlap >= self._config.ngram_threshold
        return ContaminationResult(
            is_contaminated=is_contaminated,
            method=f"ngram_{n}",
            score=max_overlap,
            matched_sources=tuple(matched),
        )

    def check_all(self, text: str, n: int = 5) -> ContaminationResult:
        exact = self.check_exact(text)
        if exact.is_contaminated:
            return exact
        norm = self.check_normalized(text)
        if norm.is_contaminated:
            return norm
        return self.check_ngram(text, n)

    def reset(self) -> None:
        self._entries.clear()
        self._entry_set.clear()
        self._exact_hashes.clear()
        self._normalized_set.clear()
        self._ngram_cache.clear()
