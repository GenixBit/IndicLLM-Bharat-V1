from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from bharat.data.normalization import NormalizationConfig, Normalizer


@dataclass(frozen=True)
class ExactDedupConfig:
    normalize: bool = True
    line_level: bool = False
    hash_func: str = "sha256"
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)


class ExactDeduplicator:
    def __init__(self, config: ExactDedupConfig | None = None) -> None:
        self.config = config or ExactDedupConfig()
        if self.config.hash_func not in hashlib.algorithms_available:
            raise ValueError(
                f"Unknown hash algorithm: {self.config.hash_func!r}. "
                f"Choose from: {sorted(hashlib.algorithms_available)}"
            )
        self._seen: set[str] = set()
        if self.config.normalize:
            normalizer = Normalizer(self.config.normalization_config)
            self._normalize_fn: Callable[[str], str] = normalizer.normalize
        else:
            self._normalize_fn = lambda x: x

    def add_document(self, text: str) -> bool:
        if not text:
            return False
        if self.config.line_level:
            return self._add_lines(text)
        key = self._make_key(text)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def is_duplicate(self, text: str) -> bool:
        if not text:
            return False
        if self.config.line_level:
            lines = text.splitlines()
            return any(self._make_key(line) in self._seen for line in lines if line.strip())
        return self._make_key(text) in self._seen

    def filter(self, texts: list[str]) -> list[str]:
        result: list[str] = []
        for text in texts:
            if self.add_document(text):
                result.append(text)
        return result

    def reset(self) -> None:
        self._seen.clear()

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def _make_key(self, text: str) -> str:
        normalized = self._normalize_fn(text)
        h = hashlib.new(self.config.hash_func)
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()

    def _add_lines(self, text: str) -> bool:
        added = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            key = self._make_key(stripped)
            if key not in self._seen:
                self._seen.add(key)
                added = True
        return added
