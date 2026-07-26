from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bharat.serving.gguf_quant_writer import BLOCK_Q8_0_SIZE, QK8_0
from bharat.serving.gguf_writer import GGUFTensorInventoryEntry

GGML_TYPE_Q8_0 = 8


@dataclass(frozen=True)
class GGUFQ8TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    element_count: int
    payload_bytes: int


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def q8_0_payload_size(shape: tuple[int, ...]) -> int:
    """Return the exact Q8_0 payload size for a tensor shape.

    Q8_0 encodes fixed blocks of 32 values into 34 bytes. Partial blocks are
    intentionally rejected because the merged quantizer accepts complete blocks only.
    """
    element_count = _element_count(shape)
    if element_count % QK8_0 != 0:
        raise ValueError(
            f"Q8_0 tensor element count ({element_count}) must be a multiple of {QK8_0}"
        )
    return (element_count // QK8_0) * BLOCK_Q8_0_SIZE


def build_q8_0_tensor_descriptors(
    tensors: Iterable[GGUFTensorInventoryEntry],
    *,
    alignment: int,
) -> tuple[GGUFQ8TensorDescriptor, ...]:
    """Build deterministic uniform-Q8_0 descriptors without serializing payloads."""
    if alignment <= 0 or alignment & (alignment - 1) != 0:
        raise ValueError("alignment must be a positive power of two")

    entries = sorted(tensors, key=lambda tensor: tensor.name)
    names = [tensor.name for tensor in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate tensor names are not allowed")

    descriptors: list[GGUFQ8TensorDescriptor] = []
    offset = 0
    for entry in entries:
        element_count = _element_count(entry.shape)
        payload_bytes = q8_0_payload_size(entry.shape)
        descriptors.append(
            GGUFQ8TensorDescriptor(
                name=entry.name,
                shape=entry.shape,
                ggml_type=GGML_TYPE_Q8_0,
                offset=offset,
                element_count=element_count,
                payload_bytes=payload_bytes,
            )
        )
        offset = _align(offset + payload_bytes, alignment)

    return tuple(descriptors)
