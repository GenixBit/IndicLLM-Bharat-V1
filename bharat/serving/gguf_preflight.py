from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export_inventory import CheckpointInventory

_ALLOWED_VALUE_TYPES = frozenset({"bool", "float", "int", "string"})


@dataclass(frozen=True)
class GGUFMetadataEntry:
    key: str
    value_type: str

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "value_type": self.value_type}


@dataclass(frozen=True)
class GGUFPreflightResult:
    schema_version: int
    architecture: str
    alignment: int
    tensor_count: int
    output_file: str
    metadata: tuple[GGUFMetadataEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "alignment": self.alignment,
            "tensor_count": self.tensor_count,
            "output_file": self.output_file,
            "metadata": [entry.to_dict() for entry in self.metadata],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _parse_metadata_entry(value: object) -> GGUFMetadataEntry:
    item = _require_mapping(value, "metadata entry")
    key = item.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("metadata key must be non-empty")

    value_type = item.get("value_type")
    if not isinstance(value_type, str) or value_type not in _ALLOWED_VALUE_TYPES:
        raise ValueError(f"metadata {key!r} has unsupported value_type {value_type!r}")

    return GGUFMetadataEntry(key=key, value_type=value_type)


def validate_gguf_preflight(
    inventory: CheckpointInventory,
    metadata_path: Path,
) -> GGUFPreflightResult:
    """Validate local GGUF export metadata without reading tensor payloads."""
    if not metadata_path.exists():
        raise ValueError(f"metadata path does not exist: {metadata_path}")
    if not metadata_path.is_file():
        raise ValueError(f"metadata path must be a file: {metadata_path}")

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata is not valid JSON: {metadata_path}") from exc

    root = _require_mapping(payload, "metadata")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    architecture = root.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("architecture must be non-empty")

    alignment = root.get("alignment")
    if (
        not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or alignment <= 0
        or alignment & (alignment - 1) != 0
    ):
        raise ValueError("alignment must be a positive power of two")

    tensor_count = root.get("tensor_count")
    if not isinstance(tensor_count, int) or isinstance(tensor_count, bool) or tensor_count < 0:
        raise ValueError("tensor_count must be a non-negative integer")

    output_file = root.get("output_file")
    if not isinstance(output_file, str) or not output_file.strip():
        raise ValueError("output_file must be non-empty")
    if not output_file.lower().endswith(".gguf"):
        raise ValueError("output_file must end with .gguf")

    inventory_paths = {item.relative_path for item in inventory.files}
    if output_file not in inventory_paths:
        raise ValueError(f"GGUF metadata references missing output file: {output_file}")

    metadata_value = root.get("metadata")
    if not isinstance(metadata_value, list):
        raise ValueError("metadata must be a list")
    metadata = tuple(
        sorted(
            (_parse_metadata_entry(item) for item in metadata_value),
            key=lambda item: item.key,
        )
    )
    keys = [entry.key for entry in metadata]
    if len(keys) != len(set(keys)):
        raise ValueError("metadata keys must be unique")

    return GGUFPreflightResult(
        schema_version=1,
        architecture=architecture,
        alignment=alignment,
        tensor_count=tensor_count,
        output_file=output_file,
        metadata=metadata,
    )
