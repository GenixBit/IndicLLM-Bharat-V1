import struct
from pathlib import Path

import pytest
import torch

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_tensor_writer import build_gguf_f32_payload, write_gguf_f32_tensors
from bharat.serving.gguf_writer import (
    GGUFTensorInventoryEntry,
    build_gguf_header_and_descriptors,
    build_gguf_tensor_descriptors,
)


def _preflight(*, tensor_count: int = 2) -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=tensor_count,
        output_file="bharat.gguf",
        metadata=(GGUFMetadataEntry("general.name", "string", "Bharat"),),
    )


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "z.weight": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        "a.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    }


def _prefix_size() -> int:
    tensors = _tensors()
    inventory = tuple(
        GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
        for name, tensor in sorted(tensors.items())
    )
    descriptors = build_gguf_tensor_descriptors(inventory, alignment=32)
    return len(build_gguf_header_and_descriptors(_preflight(), descriptors))


def test_build_payload_is_deterministic_and_orders_tensors_by_name() -> None:
    first = build_gguf_f32_payload(_preflight(), _tensors())
    second = build_gguf_f32_payload(_preflight(), dict(reversed(tuple(_tensors().items()))))

    assert first == second
    prefix_size = _prefix_size()
    assert struct.unpack_from("<4f", first, prefix_size) == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert struct.unpack_from("<2f", first, prefix_size + 32) == pytest.approx((5.0, 6.0))


def test_write_creates_complete_local_gguf_file(tmp_path: Path) -> None:
    output = tmp_path / "bharat.gguf"

    result = write_gguf_f32_tensors(_preflight(), _tensors(), output)

    assert output.read_bytes() == build_gguf_f32_payload(_preflight(), _tensors())
    assert result.output_path == output
    assert result.bytes_written == output.stat().st_size
    assert result.tensor_count == 2
    assert result.metadata_count == 1
    assert result.tensor_data_start_offset == _prefix_size()
    assert result.tensor_data_bytes == output.stat().st_size - _prefix_size()
    assert result.payload_complete is True


def test_tensor_count_must_match_preflight() -> None:
    with pytest.raises(ValueError, match="does not match"):
        build_gguf_f32_payload(_preflight(tensor_count=1), _tensors())


def test_only_dense_float32_tensors_are_supported() -> None:
    with pytest.raises(ValueError, match="torch.float32"):
        build_gguf_f32_payload(
            _preflight(tensor_count=1),
            {"weight": torch.tensor([1.0], dtype=torch.float64)},
        )

    sparse = torch.sparse_coo_tensor(indices=[[0]], values=[1.0], size=[1])
    with pytest.raises(ValueError, match="dense strided"):
        build_gguf_f32_payload(_preflight(tensor_count=1), {"weight": sparse})


def test_existing_output_is_not_replaced(tmp_path: Path) -> None:
    output = tmp_path / "bharat.gguf"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        write_gguf_f32_tensors(_preflight(), _tensors(), output)

    assert output.read_bytes() == b"existing"


def test_tensor_inputs_are_detached_and_normalized_without_mutation() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    view = source.t()

    payload = build_gguf_f32_payload(_preflight(tensor_count=1), {"weight": view})

    assert source.requires_grad is True
    assert view.is_contiguous() is True
    assert struct.unpack_from("<2f", payload, len(payload) - 32)[:2] == pytest.approx((1.0, 2.0))
