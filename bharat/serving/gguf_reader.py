from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.gguf_writer import GGML_TYPE_F32

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


@dataclass(frozen=True)
class GGUFReadTensor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass(frozen=True)
class GGUFReadResult:
    path: Path
    version: int
    metadata: tuple[tuple[str, bool | float | int | str], ...]
    tensors: tuple[GGUFReadTensor, ...]
    tensor_data_start_offset: int
    file_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "version": self.version,
            "metadata": dict(self.metadata),
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "ggml_type": tensor.ggml_type,
                    "offset": tensor.offset,
                }
                for tensor in self.tensors
            ],
            "tensor_data_start_offset": self.tensor_data_start_offset,
            "file_size": self.file_size,
        }


class _Cursor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.payload):
            raise ValueError("truncated GGUF payload")
        value = self.payload[self.offset : end]
        self.offset = end
        return value

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        size = self.unpack("<Q")
        try:
            return self.read(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 string in GGUF payload") from exc


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _read_metadata_value(
    cursor: _Cursor,
    value_type: int,
) -> bool | float | int | str:
    if value_type == _GGUF_TYPE_BOOL:
        return bool(cursor.unpack("<?"))
    if value_type == _GGUF_TYPE_STRING:
        return cursor.string()
    if value_type == _GGUF_TYPE_INT64:
        return int(cursor.unpack("<q"))
    if value_type == _GGUF_TYPE_FLOAT64:
        return float(cursor.unpack("<d"))
    raise ValueError(f"unsupported GGUF metadata value type: {value_type}")


def read_gguf_subset(path: Path, *, alignment: int = 32) -> GGUFReadResult:
    """Read the deterministic GGUF v3 subset emitted by this repository.

    This local-only reader validates the header, scalar metadata, F32 tensor
    descriptors, alignment, descriptor ordering, offsets, and file bounds. It
    does not interpret tensor values or claim compatibility with every GGUF
    feature.
    """
    if alignment <= 0 or alignment & (alignment - 1) != 0:
        raise ValueError("alignment must be a positive power of two")
    if path.suffix.lower() != ".gguf":
        raise ValueError("path must end with .gguf")
    if not path.exists() or not path.is_file():
        raise ValueError(f"GGUF path must exist and be a regular file: {path}")

    payload = path.read_bytes()
    cursor = _Cursor(payload)
    if cursor.read(4) != _GGUF_MAGIC:
        raise ValueError("invalid GGUF magic")
    version = cursor.unpack("<I")
    if version != _GGUF_VERSION:
        raise ValueError(f"unsupported GGUF version: {version}")
    tensor_count = cursor.unpack("<Q")
    metadata_count = cursor.unpack("<Q")

    metadata: list[tuple[str, bool | float | int | str]] = []
    metadata_keys: set[str] = set()
    for _ in range(metadata_count):
        key = cursor.string()
        if not key or key in metadata_keys:
            raise ValueError("GGUF metadata keys must be unique and non-empty")
        metadata_keys.add(key)
        value_type = cursor.unpack("<I")
        metadata.append((key, _read_metadata_value(cursor, value_type)))
    if [key for key, _ in metadata] != sorted(metadata_keys):
        raise ValueError("GGUF metadata keys are not deterministically sorted")

    tensors: list[GGUFReadTensor] = []
    tensor_names: set[str] = set()
    previous_offset = -1
    for _ in range(tensor_count):
        name = cursor.string()
        if not name or name in tensor_names:
            raise ValueError("GGUF tensor names must be unique and non-empty")
        tensor_names.add(name)
        rank = cursor.unpack("<I")
        if rank == 0:
            raise ValueError(f"tensor {name!r} rank must be positive")
        encoded_shape = tuple(cursor.unpack("<Q") for _ in range(rank))
        shape = tuple(reversed(encoded_shape))
        if any(dim == 0 for dim in shape):
            raise ValueError(f"tensor {name!r} dimensions must be positive")
        ggml_type = cursor.unpack("<I")
        if ggml_type != GGML_TYPE_F32:
            raise ValueError(f"unsupported GGML tensor type: {ggml_type}")
        offset = cursor.unpack("<Q")
        if offset % alignment != 0 or offset <= previous_offset:
            raise ValueError("GGUF tensor offsets must be increasing and aligned")
        previous_offset = offset
        tensors.append(
            GGUFReadTensor(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                offset=offset,
            )
        )
    if [tensor.name for tensor in tensors] != sorted(tensor_names):
        raise ValueError("GGUF tensor names are not deterministically sorted")

    data_start = _align(cursor.offset, alignment)
    if data_start > len(payload):
        raise ValueError("truncated GGUF alignment padding")
    if any(payload[cursor.offset : data_start]):
        raise ValueError("GGUF alignment padding must be zero-filled")

    relative_payload_end = 0
    for tensor in tensors:
        element_count = 1
        for dim in tensor.shape:
            element_count *= dim
        tensor_end = tensor.offset + element_count * 4
        relative_payload_end = max(relative_payload_end, tensor_end)
        if data_start + tensor_end > len(payload):
            raise ValueError(f"tensor {tensor.name!r} payload exceeds file bounds")

    expected_file_size = data_start + _align(relative_payload_end, alignment)
    if len(payload) < expected_file_size:
        raise ValueError("GGUF tensor payload exceeds file bounds or final padding is truncated")
    if len(payload) > expected_file_size:
        raise ValueError("GGUF payload contains trailing bytes")
    final_padding_start = data_start + relative_payload_end
    if any(payload[final_padding_start:expected_file_size]):
        raise ValueError("GGUF final alignment padding must be zero-filled")

    return GGUFReadResult(
        path=path.resolve(),
        version=version,
        metadata=tuple(metadata),
        tensors=tuple(tensors),
        tensor_data_start_offset=data_start,
        file_size=len(payload),
    )
