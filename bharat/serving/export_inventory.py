from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CheckpointInventory:
    checkpoint_path: Path
    files: tuple[CheckpointFile, ...]
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "files": [item.to_dict() for item in self.files],
            "total_bytes": self.total_bytes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_inventory(checkpoint_path: Path) -> CheckpointInventory:
    """Build a deterministic local inventory without loading model weights."""
    if not checkpoint_path.exists():
        raise ValueError(f"checkpoint path does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise ValueError(f"checkpoint path must be a directory: {checkpoint_path}")

    files: list[CheckpointFile] = []
    for path in sorted(item for item in checkpoint_path.rglob("*") if item.is_file()):
        relative_path = path.relative_to(checkpoint_path).as_posix()
        size_bytes = path.stat().st_size
        files.append(
            CheckpointFile(
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=_sha256(path),
            )
        )

    if not files:
        raise ValueError(f"checkpoint directory contains no files: {checkpoint_path}")

    return CheckpointInventory(
        checkpoint_path=checkpoint_path,
        files=tuple(files),
        total_bytes=sum(item.size_bytes for item in files),
    )
