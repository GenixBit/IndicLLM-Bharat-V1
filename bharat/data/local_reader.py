from __future__ import annotations

import json as json_lib
from pathlib import Path

from bharat.data.records import RawRecord


def read_local_text(path: str | Path) -> list[RawRecord]:
    raw = str(path)
    if "://" in raw:
        raise ValueError(f"Remote URLs are not supported: {raw}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {p}")
    if p.suffix == ".jsonl":
        return _read_jsonl(p)
    return _read_txt(p)


def _read_txt(path: Path) -> list[RawRecord]:
    text = path.read_text(encoding="utf-8")
    return [
        RawRecord(
            source_path=str(path),
            line_number=1,
            text=text,
        )
    ]


def _read_jsonl(path: Path) -> list[RawRecord]:
    records: list[RawRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json_lib.loads(stripped)
            except json_lib.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {lineno}: {e}") from e
            if isinstance(obj, dict) and "text" in obj:
                text = obj["text"]
            else:
                text = stripped
            records.append(
                RawRecord(
                    source_path=str(path),
                    line_number=lineno,
                    text=text,
                )
            )
    if not records:
        raise ValueError(f"Empty JSONL file: {path}")
    return records
