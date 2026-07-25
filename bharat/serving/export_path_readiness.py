from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export import ExportPlan


@dataclass(frozen=True)
class ExportPathReadiness:
    checkpoint_path: Path
    output_path: Path
    manifest_path: Path | None
    metadata_paths: tuple[Path, ...]
    ready: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "output_path": str(self.output_path),
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "metadata_paths": [str(path) for path in self.metadata_paths],
            "ready": self.ready,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def validate_export_path_readiness(
    plan: ExportPlan,
    *,
    manifest_path: Path | None = None,
    metadata_paths: Iterable[Path] = (),
) -> ExportPathReadiness:
    """Validate that local export inputs and targets cannot overwrite one another."""
    checkpoint_path = plan.checkpoint_path.resolve()
    output_path = plan.output_path.resolve()
    resolved_manifest = None if manifest_path is None else manifest_path.resolve()
    resolved_metadata = tuple(sorted((path.resolve() for path in metadata_paths), key=str))

    if len(set(resolved_metadata)) != len(resolved_metadata):
        raise ValueError("metadata paths must be unique")

    for metadata_path in resolved_metadata:
        if not metadata_path.exists():
            raise ValueError(f"metadata path does not exist: {metadata_path}")
        if not metadata_path.is_file():
            raise ValueError(f"metadata path must be a file: {metadata_path}")
        if metadata_path == output_path:
            raise ValueError("metadata path must not equal export output path")
        if resolved_manifest is not None and metadata_path == resolved_manifest:
            raise ValueError("metadata path must not equal export manifest path")

    return ExportPathReadiness(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        manifest_path=resolved_manifest,
        metadata_paths=resolved_metadata,
    )
