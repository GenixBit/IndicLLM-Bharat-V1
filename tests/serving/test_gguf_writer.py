import struct
from pathlib import Path

import pytest

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_writer import build_gguf_header, write_gguf_header


def _preflight(*, tensor_count: int = 0) -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=tensor_count,
        output_file="bharat.gguf",
        metadata=(
            GGUFMetadataEntry("general.name", "string", "Bharat"),
            GGUFMetadataEntry("general.scale", "float", 1.0),
            GGUFMetadataEntry("general.quantized", "bool", False),
            GGUFMetadataEntry("general.file_type", "int", 1),
        ),
    )


def test_build_gguf_header_is_deterministic_and_has_v3_header() -> None:
    first = build_gguf_header(_preflight())
    second = build_gguf_header(_preflight())

    assert first == second
    assert first[:4] == b"GGUF"
    assert struct.unpack_from("<I", first, 4)[0] == 3
    assert struct.unpack_from("<Q", first, 8)[0] == 0
    assert struct.unpack_from("<Q", first, 16)[0] == 4


def test_metadata_is_encoded_in_sorted_key_order() -> None:
    payload = build_gguf_header(_preflight())

    positions = [
        payload.index(key.encode("utf-8"))
        for key in (
            "general.file_type",
            "general.name",
            "general.quantized",
            "general.scale",
        )
    ]
    assert positions == sorted(positions)


def test_write_gguf_header_creates_local_file(tmp_path: Path) -> None:
    output = tmp_path / "bharat.gguf"

    result = write_gguf_header(_preflight(), output)

    assert output.read_bytes() == build_gguf_header(_preflight())
    assert result.output_path == output
    assert result.bytes_written == output.stat().st_size
    assert result.metadata_count == 4
    assert result.tensor_count == 0


def test_existing_output_is_not_replaced(tmp_path: Path) -> None:
    output = tmp_path / "bharat.gguf"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        write_gguf_header(_preflight(), output)

    assert output.read_bytes() == b"existing"


def test_nonzero_tensor_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="tensor_count to be 0"):
        build_gguf_header(_preflight(tensor_count=1))


def test_signed_64_bit_integer_range_is_enforced() -> None:
    preflight = GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=0,
        output_file="bharat.gguf",
        metadata=(GGUFMetadataEntry("general.value", "int", 2**63),),
    )

    with pytest.raises(ValueError, match="outside signed 64-bit range"):
        build_gguf_header(preflight)
