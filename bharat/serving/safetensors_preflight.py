from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export_inventory import CheckpointInventory

_ALLOWED_DTYPES = frozenset(
    {
        "BF16",
        "F16",
        "F32",
        "F64",
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "BOOL",
    }
)


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "shard": self.shard,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class SafetensorsPreflightResult:
    schema_version: int
    tensor_count: int
    total_tensor_bytes: int
    tensors: tuple[TensorMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_count": self.tensor_count,
            "total_tensor_bytes": self.total_tensor_bytes,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _parse_tensor(value: object) -> TensorMetadata:
    item = _require_mapping(value, "tensor")

    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tensor name must be non-empty")

    shape_value = item.get("shape")
    if not isinstance(shape_value, list) or not shape_value:
        raise ValueError(f"tensor {name!r} shape must be a non-empty list")
    invalid_dimension = any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in shape_value
    )
    if invalid_dimension:
        raise ValueError(f"tensor {name!r} shape dimensions must be positive integers")
    shape = tuple(int(dimension) for dimension in shape_value)

    dtype = item.get("dtype")
    if not isinstance(dtype, str) or dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"tensor {name!r} has unsupported dtype {dtype!r}")

    shard = item.get("shard")
    if not isinstance(shard, str) or not shard.strip():
        raise ValueError(f"tensor {name!r} shard must be non-empty")

    size_bytes = item.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ValueError(f"tensor {name!r} size_bytes must be a non-negative integer")

    return TensorMetadata(
        name=name,
        shape=shape,
        dtype=dtype,
        shard=shard,
        size_bytes=size_bytes,
    )


def validate_safetensors_preflight(
    inventory: CheckpointInventory,
    metadata_path: Path,
) -> SafetensorsPreflightResult:
    """Validate local safetensors metadata without reading tensor payloads."""
    if not metadata_path.exists():
        raise ValueError(f"metadata path does not exist: {metadata_path}")
    if not metadata_path.is_file():
        raise ValueError(f"metadata path must be a file: {metadata_path}")

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata is not valid JSON: {metadata_path}") from exc

    root = _require_mapping(payload, "metadata")
    schema_version = root.get("schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")

    tensors_value = root.get("tensors")
    if not isinstance(tensors_value, list) or not tensors_value:
        raise ValueError("tensors must be a non-empty list")

    tensors = tuple(
        sorted(
            (_parse_tensor(item) for item in tensors_value),
            key=lambda item: item.name,
        )
    )
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise ValueError("tensor names must be unique")

    inventory_paths = {item.relative_path for item in inventory.files}
    missing_shards = sorted(
        {tensor.shard for tensor in tensors if tensor.shard not in inventory_paths}
    )
    if missing_shards:
        missing = ", ".join(missing_shards)
        raise ValueError(f"tensor metadata references missing shards: {missing}")

    total_tensor_bytes = sum(tensor.size_bytes for tensor in tensors)
    declared_total = root.get("total_tensor_bytes")
    if (
        not isinstance(declared_total, int)
        or isinstance(declared_total, bool)
        or declared_total < 0
    ):
        raise ValueError("total_tensor_bytes must be a non-negative integer")
    if declared_total != total_tensor_bytes:
        raise ValueError(
            f"declared total_tensor_bytes {declared_total} "
            f"does not match tensor sum {total_tensor_bytes}"
        )

    return SafetensorsPreflightResult(
        schema_version=1,
        tensor_count=len(tensors),
        total_tensor_bytes=total_tensor_bytes,
        tensors=tensors,
    )
