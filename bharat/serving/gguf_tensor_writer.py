from __future__ import annotations

import os
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bharat.serving.gguf_preflight import GGUFPreflightResult
from bharat.serving.gguf_quant_writer import QK8_0, quantize_q8_0
from bharat.serving.gguf_writer import (
    GGML_TYPE_Q8_0,
    GGUFTensorInventoryEntry,
    build_gguf_header_and_descriptors,
    build_gguf_tensor_descriptors,
)


@dataclass(frozen=True)
class GGUFTensorWriteResult:
    output_path: Path
    bytes_written: int
    metadata_count: int
    tensor_count: int
    alignment: int
    tensor_data_start_offset: int
    tensor_data_bytes: int
    payload_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "bytes_written": self.bytes_written,
            "metadata_count": self.metadata_count,
            "tensor_count": self.tensor_count,
            "alignment": self.alignment,
            "tensor_data_start_offset": self.tensor_data_start_offset,
            "tensor_data_bytes": self.tensor_data_bytes,
            "payload_complete": self.payload_complete,
        }


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _normalize_f32_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not tensors:
        raise ValueError("tensors must not be empty")

    normalized: dict[str, torch.Tensor] = {}
    for name in sorted(tensors):
        if not name:
            raise ValueError("tensor names must not be empty")
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor {name!r} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"tensor {name!r} must use dense strided layout")
        if tensor.dtype != torch.float32:
            raise ValueError(f"tensor {name!r} must use torch.float32")
        if tensor.ndim == 0:
            raise ValueError(f"tensor {name!r} must have at least one dimension")
        if any(dim <= 0 for dim in tensor.shape):
            raise ValueError(f"tensor {name!r} dimensions must be positive")
        normalized[name] = tensor.detach().to(device="cpu").contiguous()
    return normalized


def _validate_q8_0_shapes(tensors: dict[str, torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        total = tensor.numel()
        if total % QK8_0 != 0:
            raise ValueError(
                f"Q8_0 requires tensor {name!r} element count ({total}) "
                f"to be a multiple of {QK8_0}"
            )


def build_gguf_q8_0_payload(
    preflight: GGUFPreflightResult,
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    """Build a deterministic GGUF v3 payload with Q8_0 quantized tensor data."""
    normalized = _normalize_f32_tensors(tensors)
    if preflight.tensor_count != len(normalized):
        raise ValueError(
            f"tensor mapping count {len(normalized)} does not match "
            f"preflight tensor_count {preflight.tensor_count}"
        )
    _validate_q8_0_shapes(normalized)

    inventory = tuple(
        GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
        for name, tensor in normalized.items()
    )
    descriptors = build_gguf_tensor_descriptors(
        inventory, alignment=preflight.alignment, ggml_type=GGML_TYPE_Q8_0
    )
    prefix = build_gguf_header_and_descriptors(preflight, descriptors)

    parts = [prefix]
    relative_offset = 0
    for descriptor in descriptors:
        if descriptor.offset < relative_offset:
            raise ValueError(f"tensor {descriptor.name!r} descriptor offset overlaps prior payload")
        padding = descriptor.offset - relative_offset
        if padding:
            parts.append(b"\x00" * padding)

        tensor = normalized[descriptor.name]
        values = tensor.reshape(-1).tolist()
        tensor_bytes = quantize_q8_0(values)
        parts.append(tensor_bytes)
        relative_offset = descriptor.offset + len(tensor_bytes)

    final_padding = _align(relative_offset, preflight.alignment) - relative_offset
    if final_padding:
        parts.append(b"\x00" * final_padding)
    return b"".join(parts)


def write_gguf_q8_0_tensors(
    preflight: GGUFPreflightResult,
    tensors: Mapping[str, torch.Tensor],
    output_path: Path,
) -> GGUFTensorWriteResult:
    """Write a complete local GGUF v3 file with Q8_0 quantized tensors."""
    payload = build_gguf_q8_0_payload(preflight, tensors)
    normalized = _normalize_f32_tensors(tensors)
    inventory = tuple(
        GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
        for name, tensor in normalized.items()
    )
    descriptors = build_gguf_tensor_descriptors(
        inventory, alignment=preflight.alignment, ggml_type=GGML_TYPE_Q8_0
    )
    prefix = build_gguf_header_and_descriptors(preflight, descriptors)
    _write_payload(payload, output_path)

    return GGUFTensorWriteResult(
        output_path=output_path,
        bytes_written=len(payload),
        metadata_count=len(preflight.metadata),
        tensor_count=len(normalized),
        alignment=preflight.alignment,
        tensor_data_start_offset=len(prefix),
        tensor_data_bytes=len(payload) - len(prefix),
    )


def build_gguf_f32_payload(
    preflight: GGUFPreflightResult,
    tensors: Mapping[str, torch.Tensor],
) -> bytes:
    """Build a deterministic GGUF v3 payload for local dense F32 tensors."""
    normalized = _normalize_f32_tensors(tensors)
    if preflight.tensor_count != len(normalized):
        raise ValueError(
            f"tensor mapping count {len(normalized)} does not match "
            f"preflight tensor_count {preflight.tensor_count}"
        )

    inventory = tuple(
        GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
        for name, tensor in normalized.items()
    )
    descriptors = build_gguf_tensor_descriptors(inventory, alignment=preflight.alignment)
    prefix = build_gguf_header_and_descriptors(preflight, descriptors)

    parts = [prefix]
    relative_offset = 0
    for descriptor in descriptors:
        if descriptor.offset < relative_offset:
            raise ValueError(f"tensor {descriptor.name!r} descriptor offset overlaps prior payload")
        padding = descriptor.offset - relative_offset
        if padding:
            parts.append(b"\x00" * padding)

        tensor = normalized[descriptor.name]
        values = tensor.reshape(-1).tolist()
        tensor_bytes = struct.pack(f"<{len(values)}f", *values)
        parts.append(tensor_bytes)
        relative_offset = descriptor.offset + len(tensor_bytes)

    final_padding = _align(relative_offset, preflight.alignment) - relative_offset
    if final_padding:
        parts.append(b"\x00" * final_padding)
    return b"".join(parts)


def _write_payload(payload: bytes, output_path: Path) -> None:
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


def write_gguf_f32_tensors(
    preflight: GGUFPreflightResult,
    tensors: Mapping[str, torch.Tensor],
    output_path: Path,
) -> GGUFTensorWriteResult:
    """Write a complete local GGUF v3 file for dense unquantized F32 tensors."""
    payload = build_gguf_f32_payload(preflight, tensors)
    normalized = _normalize_f32_tensors(tensors)
    inventory = tuple(
        GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
        for name, tensor in normalized.items()
    )
    descriptors = build_gguf_tensor_descriptors(inventory, alignment=preflight.alignment)
    prefix = build_gguf_header_and_descriptors(preflight, descriptors)
    _write_payload(payload, output_path)

    return GGUFTensorWriteResult(
        output_path=output_path,
        bytes_written=len(payload),
        metadata_count=len(preflight.metadata),
        tensor_count=len(normalized),
        alignment=preflight.alignment,
        tensor_data_start_offset=len(prefix),
        tensor_data_bytes=len(payload) - len(prefix),
    )
