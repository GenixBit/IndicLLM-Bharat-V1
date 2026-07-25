from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export import ExportPlan


@dataclass(frozen=True)
class ExportManifestReadiness:
    manifest_path: Path
    manifest_parent: Path
    output_path: Path
    checkpoint_path: Path
    manifest_exists: bool = False
    manifest_conflicts_with_output: bool = False
    manifest_inside_checkpoint: bool = False
    ready: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_parent": str(self.manifest_parent),
            "output_path": str(self.output_path),
            "checkpoint_path": str(self.checkpoint_path),
            "manifest_exists": self.manifest_exists,
            "manifest_conflicts_with_output": self.manifest_conflicts_with_output,
            "manifest_inside_checkpoint": self.manifest_inside_checkpoint,
            "ready": self.ready,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_export_manifest_readiness(
    plan: ExportPlan,
    manifest_path: Path,
) -> ExportManifestReadiness:
    """Validate a local manifest target without creating or modifying files."""
    checkpoint_path = plan.checkpoint_path.resolve()
    output_path = plan.output_path.resolve()
    resolved_manifest_path = manifest_path.resolve()
    manifest_parent = resolved_manifest_path.parent

    if not manifest_parent.exists():
        raise ValueError(f"manifest parent directory does not exist: {manifest_parent}")
    if not manifest_parent.is_dir():
        raise ValueError(f"manifest parent path must be a directory: {manifest_parent}")
    if resolved_manifest_path.exists():
        raise ValueError(f"manifest path already exists: {resolved_manifest_path}")
    if resolved_manifest_path == output_path:
        raise ValueError("manifest path must not equal export output path")
    if _is_relative_to(resolved_manifest_path, checkpoint_path):
        raise ValueError("manifest path must not be inside the checkpoint directory")

    return ExportManifestReadiness(
        manifest_path=resolved_manifest_path,
        manifest_parent=manifest_parent,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
