from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContaminationResult:
    is_contaminated: bool
    method: str
    score: float
    matched_sources: tuple[str, ...] = ()


class ContaminationChecker:
    def __init__(self) -> None:
        self._blocklist: set[str] = set()
        self._normalized_blocklist: set[str] = set()
        self._ngram_blocklist: dict[int, set[int]] = {}

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
            self._blocklist.add(entry)
            normalized = self._normalize(entry)
            self._normalized_blocklist.add(normalized)

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _text_ngrams(self, text: str, n: int) -> set[int]:
        words = text.split()
        if len(words) < n:
            effective = max(1, len(text))
            return {
                int(hashlib.md5(text[i : i + effective].encode("utf-8")).hexdigest()[:8], 16)
                for i in range(max(1, len(text) - effective + 1))
            }
        ngrams: set[int] = set()
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i : i + n])
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
            ngrams.add(h)
        return ngrams

    def check_exact(self, text: str) -> ContaminationResult:
        if not text or not self._blocklist:
            return ContaminationResult(False, "exact", 0.0)
        h = self._sha256(text)
        if h in {self._sha256(b) for b in self._blocklist}:
            return ContaminationResult(True, "exact", 1.0)
        return ContaminationResult(False, "exact", 0.0)

    def check_normalized(self, text: str) -> ContaminationResult:
        if not text or not self._blocklist:
            return ContaminationResult(False, "normalized", 0.0)
        normalized = self._normalize(text)
        if normalized in self._normalized_blocklist:
            return ContaminationResult(True, "normalized", 1.0)
        return ContaminationResult(False, "normalized", 0.0)

    def check_ngram(self, text: str, n: int = 5) -> ContaminationResult:
        if not text or not self._blocklist:
            return ContaminationResult(False, f"ngram_{n}", 0.0)
        text_ngs = self._text_ngrams(text, n)
        if not text_ngs:
            return ContaminationResult(False, f"ngram_{n}", 0.0)
        max_overlap = 0.0
        matched: list[str] = []
        for blocked in self._blocklist:
            block_ngs = self._text_ngrams(blocked, n)
            if not block_ngs:
                continue
            common = len(text_ngs & block_ngs)
            overlap = common / max(len(text_ngs), len(block_ngs))
            if overlap > max_overlap:
                max_overlap = overlap
            if overlap >= 0.5:
                matched.append(self._sha256(blocked)[:12])
        is_contaminated = max_overlap >= 0.5
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
        self._blocklist.clear()
        self._normalized_blocklist.clear()
        self._ngram_blocklist.clear()
