from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REMOTE_RE = re.compile(r"^(?:https?|ftp|s3|gs)://", re.IGNORECASE)


@dataclass(frozen=True)
class TokenizerCorpusSample:
    seed: int
    requested_records: int
    selected_records: int
    source_release_digest: str
    corpus_sha256: str
    record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "requested_records": self.requested_records,
            "selected_records": self.selected_records,
            "source_release_digest": self.source_release_digest,
            "corpus_sha256": self.corpus_sha256,
            "record_ids": list(self.record_ids),
        }


def _require_local(path: str | Path) -> Path:
    value = str(path)
    if _REMOTE_RE.match(value):
        raise ValueError(f"remote paths are not allowed: {value}")
    return Path(value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _release_digest(release: dict[str, Any]) -> str:
    canonical = json.dumps(release, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _score(seed: int, record_id: str, text: str) -> bytes:
    payload = f"{seed}\0{record_id}\0{text}".encode()
    return hashlib.sha256(payload).digest()


def sample_tokenizer_corpus(
    release_dir: str | Path,
    output_path: str | Path,
    *,
    sample_size: int,
    seed: int = 0,
) -> TokenizerCorpusSample:
    """Create a deterministic local text corpus from an approved dataset release.

    The release directory must contain ``dataset_release.json``, ``audit_report.json``,
    and a ``shards`` directory of UTF-8 JSONL files. Only accepted records with a
    non-empty string ``text`` and ``record_id`` are eligible. The output is written
    with exclusive creation so an existing corpus is never overwritten.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    release_root = _require_local(release_dir)
    destination = _require_local(output_path)
    if not release_root.is_dir():
        raise FileNotFoundError(f"release directory not found: {release_root}")

    release_path = release_root / "dataset_release.json"
    audit_path = release_root / "audit_report.json"
    shards_dir = release_root / "shards"
    if not release_path.is_file() or not audit_path.is_file() or not shards_dir.is_dir():
        raise ValueError(
            "release must contain dataset_release.json, audit_report.json, and shards/"
        )

    release = _load_json(release_path)
    audit = _load_json(audit_path)
    if (
        audit.get("approval_checks_passed") is not True
        or audit.get("shard_checks_passed") is not True
    ):
        raise ValueError("dataset release audit is not fully approved")
    if audit.get("dataset_id") != release.get("dataset_id"):
        raise ValueError("dataset release and audit dataset_id values differ")

    candidates: list[tuple[bytes, str, str]] = []
    seen_ids: set[str] = set()
    for shard_path in sorted(shards_dir.glob("*.jsonl"), key=lambda path: path.name):
        for line_number, raw_line in enumerate(
            shard_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"record must be an object: {shard_path}:{line_number}")
            if value.get("accepted", True) is not True:
                continue
            record_id = value.get("record_id")
            text = value.get("text")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"record_id must be a non-empty string: {shard_path}:{line_number}"
                )
            if record_id in seen_ids:
                raise ValueError(f"duplicate record_id: {record_id}")
            seen_ids.add(record_id)
            if not isinstance(text, str) or not text:
                raise ValueError(f"text must be a non-empty string: {shard_path}:{line_number}")
            candidates.append((_score(seed, record_id, text), record_id, text))

    if sample_size > len(candidates):
        raise ValueError(
            f"sample_size {sample_size} exceeds eligible record count {len(candidates)}"
        )

    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:sample_size]
    selected.sort(key=lambda item: item[1])
    corpus = "".join(f"{text}\n" for _, _, text in selected)
    corpus_bytes = corpus.encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(corpus_bytes)

    return TokenizerCorpusSample(
        seed=seed,
        requested_records=sample_size,
        selected_records=len(selected),
        source_release_digest=_release_digest(release),
        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        record_ids=tuple(record_id for _, record_id, _ in selected),
    )
