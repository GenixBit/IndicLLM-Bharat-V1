from __future__ import annotations

import os
import struct
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
_SUPPORTED_GGML_TYPES = frozenset({GGML_TYPE_F32})


@dataclass(frozen=True)
class GGUFWriteResult:
    output_path: Path
    bytes_written: int
    metadata_count: int
    tensor_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "output_path": str(self.output_path),
            "bytes_written": self.bytes_written,
            "metadata_count": self.metadata_count,
            "tensor_count": self.tensor_count,
        }


@dataclass(frozen=True)
class GGUFTensorInventoryEntry:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if not self.shape:
            raise ValueError(f"tensor {self.name!r} shape must not be empty")
        for i, dim in enumerate(self.shape):
            if isinstance(dim, bool):
                raise ValueError(f"tensor {self.name!r} dimension {i} must not be bool")
            if dim <= 0:
                raise ValueError(f"tensor {self.name!r} dimension {i} must be positive")
        if len(self.shape) > 2**32 - 1:
            raise ValueError(f"tensor {self.name!r} rank exceeds uint32 maximum")


@dataclass(frozen=True)
class GGUFTensorDescriptor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("descriptor name must not be empty")
        if self.ggml_type not in _SUPPORTED_GGML_TYPES:
            raise ValueError(f"unsupported GGML type: {self.ggml_type}")
        if self.offset < 0:
            raise ValueError(f"descriptor {self.name!r} offset must be non-negative")
        if self.offset > 2**64 - 1:
            raise ValueError(f"descriptor {self.name!r} offset exceeds uint64 maximum")


@dataclass(frozen=True)
class GGUFDescriptorResult:
    output_path: Path
    bytes_written: int
    metadata_count: int
    tensor_count: int
    descriptor_count: int
    alignment: int
    tensor_data_start_offset: int
    projected_tensor_data_bytes: int
    descriptor_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "bytes_written": self.bytes_written,
            "metadata_count": self.metadata_count,
            "tensor_count": self.tensor_count,
            "descriptor_count": self.descriptor_count,
            "alignment": self.alignment,
            "tensor_data_start_offset": self.tensor_data_start_offset,
            "projected_tensor_data_bytes": self.projected_tensor_data_bytes,
            "descriptor_only": self.descriptor_only,
        }


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_metadata_value(entry: GGUFMetadataEntry) -> tuple[int, bytes]:
    value = entry.value
    if entry.value_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"metadata {entry.key!r} value must be bool")
        return _GGUF_TYPE_BOOL, struct.pack("<?", value)
    if entry.value_type == "float":
        if not isinstance(value, float):
            raise ValueError(f"metadata {entry.key!r} value must be float")
        return _GGUF_TYPE_FLOAT64, struct.pack("<d", value)
    if entry.value_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"metadata {entry.key!r} value must be int")
        if value < -(2**63) or value > 2**63 - 1:
            raise ValueError(f"metadata {entry.key!r} integer is outside signed 64-bit range")
        return _GGUF_TYPE_INT64, struct.pack("<q", value)
    if entry.value_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"metadata {entry.key!r} value must be string")
        return _GGUF_TYPE_STRING, _pack_string(value)
    raise ValueError(f"metadata {entry.key!r} has unsupported value_type {entry.value_type!r}")


def _write_payload(payload: bytes, output_path: Path) -> None:
    """Write payload to a local file using atomic no-overwrite publication."""
    if output_path.suffix.lower() != ".gguf":
        raise ValueError("output_path must end with .gguf")
    if not output_path.parent.exists() or not output_path.parent.is_dir():
        raise ValueError(f"output parent must exist and be a directory: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"output path already exists: {output_path}")

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(tmp_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(f"output path was created concurrently: {output_path}") from exc
        tmp_path.unlink()
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _f32_byte_size(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total * 4


def build_gguf_header(preflight: GGUFPreflightResult) -> bytes:
    """Build a deterministic GGUF v3 header containing scalar metadata only."""
    if preflight.tensor_count != 0:
        raise ValueError("GGUF header writer currently requires tensor_count to be 0")

    metadata = tuple(sorted(preflight.metadata, key=lambda item: item.key))
    parts = [
        _GGUF_MAGIC,
        struct.pack("<I", _GGUF_VERSION),
        struct.pack("<Q", 0),
        struct.pack("<Q", len(metadata)),
    ]
    for entry in metadata:
        value_type, value_bytes = _pack_metadata_value(entry)
        parts.extend((_pack_string(entry.key), struct.pack("<I", value_type), value_bytes))
    return b"".join(parts)


def build_gguf_tensor_descriptors(
    tensors: Iterable[GGUFTensorInventoryEntry],
    *,
    alignment: int,
) -> tuple[GGUFTensorDescriptor, ...]:
    """Build deterministic tensor descriptors with computed offsets.

    Descriptors are sorted by name.  Offsets are relative to the start of
    the tensor-data section.  Each tensor payload begins at the aligned end
    of the previous payload.
    """
    if alignment <= 0 or alignment & (alignment - 1) != 0:
        raise ValueError("alignment must be a positive power of two")

    entries = sorted(tensors, key=lambda t: t.name)
    names = [t.name for t in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate tensor names are not allowed")

    descriptors: list[GGUFTensorDescriptor] = []
    offset = 0
    for entry in entries:
        byte_size = _f32_byte_size(entry.shape)
        descriptors.append(
            GGUFTensorDescriptor(
                name=entry.name,
                shape=entry.shape,
                ggml_type=GGML_TYPE_F32,
                offset=offset,
            )
        )
        offset = _align(offset + byte_size, alignment)

    return tuple(descriptors)


def build_gguf_header_and_descriptors(
    preflight: GGUFPreflightResult,
    descriptors: tuple[GGUFTensorDescriptor, ...],
) -> bytes:
    """Build a deterministic GGUF v3 file with header, metadata, and tensor descriptors.

    When *tensor_count* is zero the output is identical to :func:`build_gguf_header`.
    When *tensor_count* is non-zero the output ends at the aligned start of the
    tensor-data section (no tensor payload bytes are written).
    """
    if preflight.tensor_count == 0:
        if descriptors:
            raise ValueError("descriptors must be empty when tensor_count is 0")
        return build_gguf_header(preflight)

    if preflight.tensor_count != len(descriptors):
        raise ValueError(
            f"descriptor count {len(descriptors)} does not match "
            f"tensor_count {preflight.tensor_count}"
        )

    metadata = tuple(sorted(preflight.metadata, key=lambda item: item.key))
    parts = [
        _GGUF_MAGIC,
        struct.pack("<I", _GGUF_VERSION),
        struct.pack("<Q", preflight.tensor_count),
        struct.pack("<Q", len(metadata)),
    ]
    for entry in metadata:
        value_type, value_bytes = _pack_metadata_value(entry)
        parts.extend((_pack_string(entry.key), struct.pack("<I", value_type), value_bytes))

    for desc in descriptors:
        parts.append(_pack_string(desc.name))
        parts.append(struct.pack("<I", len(desc.shape)))
        for dim in reversed(desc.shape):
            parts.append(struct.pack("<Q", dim))
        parts.append(struct.pack("<I", desc.ggml_type))
        parts.append(struct.pack("<Q", desc.offset))

    raw = b"".join(parts)
    alignment = preflight.alignment
    padding = (alignment - (len(raw) % alignment)) % alignment
    if padding > 0:
        raw += b"\x00" * padding
    return raw


def write_gguf_header(preflight: GGUFPreflightResult, output_path: Path) -> GGUFWriteResult:
    """Write a local metadata-only GGUF v3 file without replacing existing output."""
    payload = build_gguf_header(preflight)
    _write_payload(payload, output_path)
    return GGUFWriteResult(
        output_path=output_path,
        bytes_written=len(payload),
        metadata_count=len(preflight.metadata),
        tensor_count=0,
    )


def write_gguf_header_and_descriptors(
    preflight: GGUFPreflightResult,
    descriptors: tuple[GGUFTensorDescriptor, ...],
    output_path: Path,
) -> GGUFDescriptorResult:
    """Write a local descriptor-only GGUF v3 file.

    The file contains header, metadata, tensor descriptors, and alignment
    padding up to the start of the tensor-data section.  No tensor payload
    bytes are written.
    """
    payload = build_gguf_header_and_descriptors(preflight, descriptors)
    _write_payload(payload, output_path)

    tensor_data_start_offset = len(payload)

    if descriptors:
        last = descriptors[-1]
        projected = _align(last.offset + _f32_byte_size(last.shape), preflight.alignment)
    else:
        projected = 0

    return GGUFDescriptorResult(
        output_path=output_path,
        bytes_written=len(payload),
        metadata_count=len(preflight.metadata),
        tensor_count=preflight.tensor_count,
        descriptor_count=len(descriptors),
        alignment=preflight.alignment,
        tensor_data_start_offset=tensor_data_start_offset,
        projected_tensor_data_bytes=projected,
        descriptor_only=True,
    )
