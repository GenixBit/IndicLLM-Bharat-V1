from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _make_record_id(source_path: str, line_number: int) -> str:
    raw = f"{source_path}:{line_number}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RawRecord:
    source_path: str
    line_number: int
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return _make_record_id(self.source_path, self.line_number)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "line_number": self.line_number,
            "text": self.text,
        }
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


@dataclass(frozen=True)
class ProcessedRecord:
    record_id: str
    text: str
    language: str
    quality_score: float
    source_path: str
    line_number: int
    processing_reasons: tuple[str, ...] = ()
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "language": self.language,
            "quality_score": self.quality_score,
            "source_path": self.source_path,
            "line_number": self.line_number,
            "processing_reasons": list(self.processing_reasons),
            "accepted": self.accepted,
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
