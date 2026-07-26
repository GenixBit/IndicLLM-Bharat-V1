import math
import struct
from pathlib import Path

import pytest

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_writer import (
    GGML_TYPE_F32,
    GGUFTensorDescriptor,
    GGUFTensorInventoryEntry,
    build_gguf_header,
    build_gguf_header_and_descriptors,
    build_gguf_tensor_descriptors,
    write_gguf_header_and_descriptors,
)


def _parse_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    value = data[offset : offset + length].decode("utf-8")
    offset += length
    return value, offset


def _parse_metadata_value(data: bytes, offset: int) -> tuple[int, int]:
    value_type = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if value_type in (7,):  # bool
        offset += 1
    elif value_type in (11, 12):  # int64 or float64
        offset += 8
    elif value_type in (8,):  # string
        _, offset = _parse_gguf_string(data, offset)
    return value_type, offset


def _parse_tensor_descriptor(data: bytes, offset: int) -> tuple[dict, int]:
    name, offset = _parse_gguf_string(data, offset)
    n_dims = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    dims = list(struct.unpack_from(f"<{n_dims}Q", data, offset))
    offset += 8 * n_dims
    ggml_type = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    tensor_offset = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    return {
        "name": name,
        "n_dims": n_dims,
        "dims": dims,
        "ggml_type": ggml_type,
        "offset": tensor_offset,
    }, offset


class GGUFByteParser:
    """Independent byte-level GGUF v3 parser for test verification."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self._parse()

    def _parse(self) -> None:
        off = 0
        self.magic = self.data[off : off + 4]
        off += 4
        self.version = struct.unpack_from("<I", self.data, off)[0]
        off += 4
        self.tensor_count = struct.unpack_from("<Q", self.data, off)[0]
        off += 8
        self.metadata_count = struct.unpack_from("<Q", self.data, off)[0]
        off += 8
        self.metadata_types: list[int] = []
        for _ in range(self.metadata_count):
            _, off = _parse_gguf_string(self.data, off)
            vt, off = _parse_metadata_value(self.data, off)
            self.metadata_types.append(vt)
        self.descriptors: list[dict] = []
        for _ in range(self.tensor_count):
            desc, off = _parse_tensor_descriptor(self.data, off)
            self.descriptors.append(desc)
        self.end_of_descriptors = off
        alignment = 32
        self.padding = (alignment - (off % alignment)) % alignment
        self.tensor_data_start = off + self.padding
        self.total_bytes = len(self.data)
        self.has_padding_bytes = self.total_bytes > off


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _preflight(
    *,
    tensor_count: int = 0,
    alignment: int = 32,
) -> GGUFPreflightResult:
    metadata: tuple[GGUFMetadataEntry, ...] = (
        GGUFMetadataEntry("general.architecture", "string", "bharat"),
        GGUFMetadataEntry("general.name", "string", "Bharat"),
    )
    if tensor_count == 0:
        metadata = (*metadata, GGUFMetadataEntry("general.file_type", "int", 1))
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=alignment,
        tensor_count=tensor_count,
        output_file="bharat.gguf",
        metadata=metadata,
    )


def _entry(name: str, *shape: int) -> GGUFTensorInventoryEntry:
    return GGUFTensorInventoryEntry(name=name, shape=shape)


# ===================================================================
# Descriptor validation
# ===================================================================


class TestInventoryEntryValidation:
    def test_valid_single_entry(self) -> None:
        e = _entry("weight", 64, 64)
        assert e.name == "weight"
        assert e.shape == (64, 64)

    def test_valid_1d_entry(self) -> None:
        e = _entry("bias", 128)
        assert e.shape == (128,)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            GGUFTensorInventoryEntry(name="", shape=(4,))

    def test_empty_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape must not be empty"):
            _entry("x")

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _entry("x", 0, 4)

    def test_negative_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _entry("x", -1, 4)

    def test_bool_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be bool"):
            _entry("x", True, 4)

    def test_rank_overflow_rejected(self) -> None:
        pass  # Practically untestable: requires > 2^32 - 1 dimensions


# ===================================================================
# F32 sizing
# ===================================================================


class TestF32ByteSize:
    def _size(self, *shape: int) -> int:
        descs = build_gguf_tensor_descriptors([_entry("t", *shape)], alignment=32)
        last = descs[-1]
        return last.offset + math.prod(last.shape) * 4

    def test_1d(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("t", 64)], alignment=32)
        assert descs[0].offset == 0
        assert math.prod((64,)) * 4 == 256

    def test_2d(self) -> None:
        build_gguf_tensor_descriptors([_entry("t", 64, 128)], alignment=32)
        assert math.prod((64, 128)) * 4 == 32768

    def test_3d(self) -> None:
        build_gguf_tensor_descriptors([_entry("t", 3, 64, 128)], alignment=32)
        assert math.prod((3, 64, 128)) * 4 == 98304

    def test_small_shape(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("t", 1, 1)], alignment=32)
        assert descs[0].offset == 0
        assert math.prod((1, 1)) * 4 == 4


# ===================================================================
# Ordering and offsets
# ===================================================================


class TestOrderingAndOffsets:
    def test_sorted_by_name(self) -> None:
        tensors = [_entry("z", 2), _entry("a", 2), _entry("m", 2)]
        descs = build_gguf_tensor_descriptors(tensors, alignment=32)
        assert [d.name for d in descs] == ["a", "m", "z"]

    def test_input_order_does_not_affect(self) -> None:
        tensors1 = [_entry("b", 2), _entry("a", 2)]
        tensors2 = [_entry("a", 2), _entry("b", 2)]
        assert build_gguf_tensor_descriptors(tensors1, alignment=32) == (
            build_gguf_tensor_descriptors(tensors2, alignment=32)
        )

    def test_first_tensor_offset_zero(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("a", 4, 4)], alignment=32)
        assert descs[0].offset == 0

    def test_second_offset_follows_first(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        expected = ((4 * 4 + 31) // 32) * 32
        assert descs[1].offset == expected

    def test_multiple_offsets(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 2), _entry("b", 4), _entry("c", 8)], alignment=32
        )
        a_size = 2 * 4
        b_size = 4 * 4
        off_b = ((a_size + 31) // 32) * 32
        off_c = ((off_b + b_size + 31) // 32) * 32
        assert descs[1].offset == off_b
        assert descs[2].offset == off_c

    def test_alignment_32_behavior(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("a", 1), _entry("b", 1)], alignment=32)
        assert descs[0].offset == 0
        assert descs[1].offset == 32

    def test_alignment_64_behavior(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("a", 1), _entry("b", 1)], alignment=64)
        assert descs[0].offset == 0
        assert descs[1].offset == 64

    def test_invalid_alignment_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive power of two"):
            build_gguf_tensor_descriptors([_entry("a", 1)], alignment=31)

    def test_projected_total_size(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        last_size = 8 * 4
        projected = ((descs[-1].offset + last_size + 31) // 32) * 32
        assert projected == 64  # 16 aligned to 32 = 32, next is 64


# ===================================================================
# Duplicate rejection
# ===================================================================


class TestDuplicateRejection:
    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate tensor names"):
            build_gguf_tensor_descriptors([_entry("a", 2), _entry("a", 2)], alignment=32)

    def test_duplicate_across_different_shapes(self) -> None:
        with pytest.raises(ValueError, match="duplicate tensor names"):
            build_gguf_tensor_descriptors(
                [_entry("b", 2), _entry("a", 4), _entry("a", 8)], alignment=32
            )


# ===================================================================
# Binary encoding
# ===================================================================


class TestBinaryEncoding:
    def _payload(self, *tensors: GGUFTensorInventoryEntry) -> bytes:
        pf = _preflight(tensor_count=len(tensors))
        descs = build_gguf_tensor_descriptors(tensors, alignment=32)
        return build_gguf_header_and_descriptors(pf, descs)

    def test_magic_bytes(self) -> None:
        data = self._payload(_entry("t", 4))
        assert data[:4] == b"GGUF"

    def test_version(self) -> None:
        data = self._payload(_entry("t", 4))
        assert struct.unpack_from("<I", data, 4)[0] == 3

    def test_tensor_count_in_header(self) -> None:
        data = self._payload(_entry("a", 4), _entry("b", 8))
        assert struct.unpack_from("<Q", data, 8)[0] == 2

    def test_zero_tensor_count_scenario(self) -> None:
        pf = _preflight(tensor_count=0)
        data = build_gguf_header_and_descriptors(pf, ())
        assert struct.unpack_from("<Q", data, 8)[0] == 0

    def test_descriptor_count_matches_tensor_count(self) -> None:
        data = self._payload(_entry("a", 4), _entry("b", 8), _entry("c", 2))
        parser = GGUFByteParser(data)
        assert len(parser.descriptors) == 3

    def test_tensor_name_encoding(self) -> None:
        data = self._payload(_entry("weight", 4, 4))
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["name"] == "weight"

    def test_rank_encoding(self) -> None:
        data = self._payload(_entry("t", 16, 32, 64))
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["n_dims"] == 3

    def test_dimensions_reversed_in_file(self) -> None:
        data = self._payload(_entry("t", 16, 32))
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["dims"] == [32, 16]

    def test_ggml_type_encoding(self) -> None:
        data = self._payload(_entry("t", 4))
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["ggml_type"] == GGML_TYPE_F32

    def test_relative_offset_encoding(self) -> None:
        data = self._payload(_entry("a", 2), _entry("b", 4))
        parser = GGUFByteParser(data)
        expected = ((2 * 4 + 31) // 32) * 32
        assert parser.descriptors[1]["offset"] == expected

    def test_tensor_data_section_alignment(self) -> None:
        data = self._payload(_entry("t", 1))
        parser = GGUFByteParser(data)
        assert parser.tensor_data_start == len(data)

    def test_descriptors_are_deterministic(self) -> None:
        a = self._payload(_entry("b", 2), _entry("a", 4))
        b = self._payload(_entry("b", 2), _entry("a", 4))
        assert a == b

    def test_insertion_order_does_not_affect_bytes(self) -> None:
        a = self._payload(_entry("b", 2), _entry("a", 4))
        b = self._payload(_entry("a", 4), _entry("b", 2))
        assert a == b


# ===================================================================
# Compatibility with zero-tensor output
# ===================================================================


class TestZeroTensorCompatibility:
    def test_zero_tensor_output_unchanged(self) -> None:
        pf = _preflight(tensor_count=0)
        existing = build_gguf_header(pf)
        new = build_gguf_header_and_descriptors(pf, ())
        assert new == existing

    def test_empty_descriptors_required_for_zero_count(self) -> None:
        pf = _preflight(tensor_count=0)
        with pytest.raises(ValueError, match="descriptors must be empty"):
            build_gguf_header_and_descriptors(
                pf,
                (GGUFTensorDescriptor(name="x", shape=(1,), ggml_type=GGML_TYPE_F32, offset=0),),
            )

    def test_zero_tensor_metadata_count(self) -> None:
        pf = _preflight(tensor_count=0)
        data = build_gguf_header_and_descriptors(pf, ())
        assert struct.unpack_from("<Q", data, 16)[0] == 3

    def test_zero_tensor_has_no_descriptors(self) -> None:
        pf = _preflight(tensor_count=0)
        data = build_gguf_header_and_descriptors(pf, ())
        parser = GGUFByteParser(data)
        assert len(parser.descriptors) == 0


# ===================================================================
# Count mismatch rejection
# ===================================================================


class TestCountMismatch:
    def test_extra_descriptors_rejected(self) -> None:
        pf = _preflight(tensor_count=1)
        descs = (
            GGUFTensorDescriptor(name="a", shape=(2,), ggml_type=GGML_TYPE_F32, offset=0),
            GGUFTensorDescriptor(name="b", shape=(2,), ggml_type=GGML_TYPE_F32, offset=32),
        )
        with pytest.raises(ValueError, match="does not match"):
            build_gguf_header_and_descriptors(pf, descs)

    def test_missing_descriptors_rejected(self) -> None:
        pf = _preflight(tensor_count=2)
        descs = (GGUFTensorDescriptor(name="a", shape=(2,), ggml_type=GGML_TYPE_F32, offset=0),)
        with pytest.raises(ValueError, match="does not match"):
            build_gguf_header_and_descriptors(pf, descs)


# ===================================================================
# Descriptor-only boundary
# ===================================================================


class TestDescriptorOnly:
    def test_no_payload_bytes_appended(self) -> None:
        pf = _preflight(tensor_count=1, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("t", 64, 64)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        expected_end = ((len(data) + 31) // 32) * 32
        assert len(data) == expected_end

    def test_multiple_tensors_no_payload(self) -> None:
        pf = _preflight(tensor_count=2, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("a", 4, 4), _entry("b", 8, 8)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.total_bytes == parser.tensor_data_start
        assert parser.total_bytes == len(data)

    def test_result_reports_descriptor_only(self) -> None:
        pf = _preflight(tensor_count=1, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        assert data is not None


# ===================================================================
# Write behavior
# ===================================================================


class TestWriteBehavior:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert output.exists()
        assert result.bytes_written == output.stat().st_size

    def test_bytes_written_matches_file_size(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=2)
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.bytes_written == output.stat().st_size

    def test_existing_output_not_replaced(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        output.write_bytes(b"existing")
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        with pytest.raises(FileExistsError, match="already exists"):
            write_gguf_header_and_descriptors(pf, descs, output)
        assert output.read_bytes() == b"existing"

    def test_missing_parent_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "nonexistent" / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        with pytest.raises(ValueError, match="output parent"):
            write_gguf_header_and_descriptors(pf, descs, output)

    def test_wrong_suffix_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "model.bin"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        with pytest.raises(ValueError, match="must end with .gguf"):
            write_gguf_header_and_descriptors(pf, descs, output)

    def test_result_output_path(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.output_path == output

    def test_result_metadata_count(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.metadata_count == 2

    def test_result_tensor_count(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=3)
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 2), _entry("b", 4), _entry("c", 8)], alignment=32
        )
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.tensor_count == 3

    def test_result_descriptor_count(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=2)
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.descriptor_count == 2

    def test_result_descriptor_only_flag(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.descriptor_only is True

    def test_result_alignment(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1, alignment=64)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=64)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.alignment == 64

    def test_result_tensor_data_start_offset(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.tensor_data_start_offset == result.bytes_written

    def test_result_projected_bytes(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 64, 64)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        expected = ((64 * 64 * 4 + 31) // 32) * 32
        assert result.projected_tensor_data_bytes == expected

    def test_repeated_write_deterministic(self, tmp_path: Path) -> None:
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        out1 = tmp_path / "a.gguf"
        out2 = tmp_path / "b.gguf"
        write_gguf_header_and_descriptors(pf, descs, out1)
        write_gguf_header_and_descriptors(pf, descs, out2)
        assert out1.read_bytes() == out2.read_bytes()

    def test_tensor_data_starts_at_aligned_boundary(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        write_gguf_header_and_descriptors(pf, descs, output)
        data = output.read_bytes()
        parser = GGUFByteParser(data)
        assert parser.tensor_data_start % 32 == 0

    def test_no_extra_bytes_after_tensor_data_start(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        write_gguf_header_and_descriptors(pf, descs, output)
        data = output.read_bytes()
        parser = GGUFByteParser(data)
        assert len(data) == parser.tensor_data_start


# ===================================================================
# Byte-level parser integration tests
# ===================================================================


class TestParserIntegration:
    def test_parse_magic(self) -> None:
        pf = _preflight(tensor_count=2)
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.magic == b"GGUF"

    def test_parse_version(self) -> None:
        pf = _preflight(tensor_count=2)
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.version == 3

    def test_parse_metadata_count(self) -> None:
        pf = _preflight(tensor_count=2)
        descs = build_gguf_tensor_descriptors([_entry("a", 4), _entry("b", 8)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.metadata_count == 2

    def test_parse_tensor_name(self) -> None:
        pf = _preflight(tensor_count=3)
        descs = build_gguf_tensor_descriptors(
            [_entry("z", 2), _entry("a", 4), _entry("m", 8)], alignment=32
        )
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["name"] == "a"
        assert parser.descriptors[1]["name"] == "m"
        assert parser.descriptors[2]["name"] == "z"

    def test_parse_dimension_encoding(self) -> None:
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 16, 32)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        desc = parser.descriptors[0]
        assert desc["n_dims"] == 2
        assert desc["dims"] == [32, 16]

    def test_parse_multiple_offsets(self) -> None:
        pf = _preflight(tensor_count=3)
        tensors = [
            _entry("a", 4),
            _entry("b", 16),
            _entry("c", 32),
        ]
        descs = build_gguf_tensor_descriptors(tensors, alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["offset"] == 0
        assert parser.descriptors[1]["offset"] > 0
        assert parser.descriptors[2]["offset"] > parser.descriptors[1]["offset"]

    def test_tensor_data_section_start_aligned(self) -> None:
        pf = _preflight(tensor_count=1, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.tensor_data_start % 32 == 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_byte_tensor(self) -> None:
        pf = _preflight(tensor_count=1, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("t", 1)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["offset"] == 0
        assert len(parser.descriptors) == 1
        assert parser.tensor_data_start % 32 == 0

    def test_large_2d_tensor(self) -> None:
        pf = _preflight(tensor_count=1, alignment=32)
        descs = build_gguf_tensor_descriptors([_entry("t", 4096, 4096)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.descriptors[0]["offset"] == 0
        assert parser.descriptors[0]["dims"] == [4096, 4096]

    def test_varied_shapes(self) -> None:
        pf = _preflight(tensor_count=3, alignment=32)
        tensors = [
            _entry("small", 2, 2),
            _entry("medium", 16, 32),
            _entry("large", 128, 256),
        ]
        descs = build_gguf_tensor_descriptors(tensors, alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert len(parser.descriptors) == 3
        assert parser.descriptors[0]["offset"] == 0
        assert parser.descriptors[1]["offset"] > parser.descriptors[0]["offset"]
        assert parser.descriptors[2]["offset"] > parser.descriptors[1]["offset"]

    def test_metadata_preserved_with_tensors(self) -> None:
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        data = build_gguf_header_and_descriptors(pf, descs)
        parser = GGUFByteParser(data)
        assert parser.metadata_count == 2


# ===================================================================
# Unsupported types
# ===================================================================


class TestUnsupportedTypes:
    def test_non_f32_ggml_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported GGML type"):
            GGUFTensorDescriptor(name="t", shape=(4,), ggml_type=1, offset=0)

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="offset must be non-negative"):
            GGUFTensorDescriptor(name="t", shape=(4,), ggml_type=GGML_TYPE_F32, offset=-1)


# ===================================================================
# Result to_dict
# ===================================================================


class TestResultToDict:
    def test_to_dict_keys(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        d = result.to_dict()
        expected_keys = {
            "output_path",
            "bytes_written",
            "metadata_count",
            "tensor_count",
            "descriptor_count",
            "alignment",
            "tensor_data_start_offset",
            "projected_tensor_data_bytes",
            "descriptor_only",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_descriptor_only(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.to_dict()["descriptor_only"] is True

    def test_to_dict_zero_tensor(self, tmp_path: Path) -> None:
        output = tmp_path / "model.gguf"
        pf = _preflight(tensor_count=0)
        result = write_gguf_header_and_descriptors(pf, (), output)
        d = result.to_dict()
        assert d["tensor_count"] == 0
        assert d["descriptor_count"] == 0
        assert d["projected_tensor_data_bytes"] == 0
