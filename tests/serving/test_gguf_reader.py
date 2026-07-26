import struct
from pathlib import Path

import pytest
import torch

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_reader import read_gguf_subset
from bharat.serving.gguf_tensor_writer import build_gguf_f32_payload


def _preflight() -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=2,
        output_file="bharat.gguf",
        metadata=(
            GGUFMetadataEntry("general.architecture", "string", "bharat"),
            GGUFMetadataEntry("general.enabled", "bool", True),
            GGUFMetadataEntry("general.layers", "int", 2),
            GGUFMetadataEntry("general.scale", "float", 1.5),
        ),
    )


def _payload() -> bytes:
    return build_gguf_f32_payload(
        _preflight(),
        {
            "z.weight": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
            "a.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        },
    )


def test_reader_validates_writer_output_offline(tmp_path: Path) -> None:
    path = tmp_path / "bharat.gguf"
    path.write_bytes(_payload())

    result = read_gguf_subset(path)

    assert result.version == 3
    assert dict(result.metadata) == {
        "general.architecture": "bharat",
        "general.enabled": True,
        "general.layers": 2,
        "general.scale": 1.5,
    }
    assert [tensor.name for tensor in result.tensors] == ["a.weight", "z.weight"]
    assert [tensor.shape for tensor in result.tensors] == [(2, 2), (1, 2)]
    assert [tensor.offset for tensor in result.tensors] == [0, 32]
    assert result.file_size == path.stat().st_size


def test_reader_rejects_invalid_magic(tmp_path: Path) -> None:
    path = tmp_path / "invalid.gguf"
    path.write_bytes(b"NOPE" + _payload()[4:])

    with pytest.raises(ValueError, match="magic"):
        read_gguf_subset(path)


def test_reader_rejects_truncated_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated.gguf"
    path.write_bytes(_payload()[:-1])

    with pytest.raises(ValueError, match="file bounds"):
        read_gguf_subset(path)


def test_reader_rejects_unsupported_metadata_type(tmp_path: Path) -> None:
    payload = bytearray(_payload())
    first_key_length = struct.unpack_from("<Q", payload, 24)[0]
    first_type_offset = 24 + 8 + first_key_length
    struct.pack_into("<I", payload, first_type_offset, 99)
    path = tmp_path / "unsupported.gguf"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="metadata value type"):
        read_gguf_subset(path)


def test_reader_rejects_nonzero_alignment_padding(tmp_path: Path) -> None:
    payload = bytearray(_payload())
    valid_path = tmp_path / "valid.gguf"
    valid_path.write_bytes(payload)
    result = read_gguf_subset(valid_path)
    payload[result.tensor_data_start_offset - 1] = 1
    path = tmp_path / "padding.gguf"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="zero-filled"):
        read_gguf_subset(path)


def test_reader_is_local_only_and_requires_regular_gguf_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end with .gguf"):
        read_gguf_subset(tmp_path / "model.bin")
    with pytest.raises(ValueError, match="regular file"):
        read_gguf_subset(tmp_path / "missing.gguf")
