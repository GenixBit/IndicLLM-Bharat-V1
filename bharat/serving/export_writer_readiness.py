from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export import ExportPlan
from bharat.serving.export_inventory import CheckpointInventory


@dataclass(frozen=True)
class ExportWriterReadiness:
    checkpoint_path: Path
    output_path: Path
    export_format: str
    checkpoint_file_count: int
    checkpoint_total_bytes: int
    output_parent: Path
    output_exists: bool = False
    output_inside_checkpoint: bool = False
    ready: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "output_path": str(self.output_path),
            "export_format": self.export_format,
            "checkpoint_file_count": self.checkpoint_file_count,
            "checkpoint_total_bytes": self.checkpoint_total_bytes,
            "output_parent": str(self.output_parent),
            "output_exists": self.output_exists,
            "output_inside_checkpoint": self.output_inside_checkpoint,
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


def validate_export_writer_readiness(
    plan: ExportPlan,
    inventory: CheckpointInventory,
) -> ExportWriterReadiness:
    """Validate local writer preconditions without creating or modifying files."""
    checkpoint_path = plan.checkpoint_path.resolve()
    inventory_checkpoint_path = inventory.checkpoint_path.resolve()
    output_path = plan.output_path.resolve()
    output_parent = output_path.parent

    if inventory_checkpoint_path != checkpoint_path:
        raise ValueError("checkpoint inventory does not match export plan checkpoint path")
    if not inventory.files:
        raise ValueError("checkpoint inventory must contain at least one file")
    if inventory.total_bytes <= 0:
        raise ValueError("checkpoint inventory must contain at least one byte")
    expected_total = sum(item.size_bytes for item in inventory.files)
    if inventory.total_bytes != expected_total:
        raise ValueError("checkpoint inventory total_bytes does not match file sizes")
    if not output_parent.exists():
        raise ValueError(f"output parent directory does not exist: {output_parent}")
    if not output_parent.is_dir():
        raise ValueError(f"output parent path must be a directory: {output_parent}")
    if output_path.exists():
        raise ValueError(f"output path already exists: {output_path}")
    if _is_relative_to(output_path, checkpoint_path):
        raise ValueError("output path must not be inside the checkpoint directory")

    return ExportWriterReadiness(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        export_format=plan.export_format,
        checkpoint_file_count=len(inventory.files),
        checkpoint_total_bytes=inventory.total_bytes,
        output_parent=output_parent,
    )
