import math
import struct
from pathlib import Path

import pytest
import torch

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_quant_writer import BLOCK_Q8_0_SIZE, QK8_0, dequantize_q8_0
from bharat.serving.gguf_reader import read_gguf_subset
from bharat.serving.gguf_tensor_writer import (
    build_gguf_f32_payload,
    build_gguf_q8_0_payload,
    write_gguf_q8_0_tensors,
)
from bharat.serving.gguf_writer import (
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    GGUFTensorDescriptor,
    GGUFTensorInventoryEntry,
    build_gguf_header_and_descriptors,
    build_gguf_tensor_descriptors,
    write_gguf_header_and_descriptors,
)


def _preflight(
    *,
    tensor_count: int = 2,
    alignment: int = 32,
) -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=alignment,
        tensor_count=tensor_count,
        output_file="bharat.gguf",
        metadata=(
            GGUFMetadataEntry("general.name", "string", "Bharat"),
            GGUFMetadataEntry("general.scale", "float", 1.0),
        ),
    )


def _f32_tensors() -> dict[str, torch.Tensor]:
    return {
        "z.weight": torch.tensor([[float(i + 1) for i in range(32)]], dtype=torch.float32),
        "a.weight": torch.tensor([[float(i + 1) for i in range(64)]], dtype=torch.float32),
    }


def _entry(name: str, *shape: int) -> GGUFTensorInventoryEntry:
    return GGUFTensorInventoryEntry(name=name, shape=shape)


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    def test_ggml_type_q8_0_value(self) -> None:
        assert GGML_TYPE_Q8_0 == 8

    def test_q8_0_block_constants(self) -> None:
        assert QK8_0 == 32
        assert BLOCK_Q8_0_SIZE == 34

    def test_q8_0_is_distinct_from_f32(self) -> None:
        assert GGML_TYPE_Q8_0 != GGML_TYPE_F32


# ===================================================================
# Q8_0 descriptor validation
# ===================================================================


class TestQ80DescriptorConstruction:
    def test_q8_0_descriptor_created(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("t", 32)], alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        assert descs[0].ggml_type == GGML_TYPE_Q8_0

    def test_f32_default_type_unchanged(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("t", 32)], alignment=32)
        assert descs[0].ggml_type == GGML_TYPE_F32

    def test_explicit_f32_type(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("t", 32)], alignment=32, ggml_type=GGML_TYPE_F32
        )
        assert descs[0].ggml_type == GGML_TYPE_F32

    def test_unsupported_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported GGML type"):
            build_gguf_tensor_descriptors([_entry("t", 32)], alignment=32, ggml_type=99)

    def test_descriptor_direct_construction(self) -> None:
        desc = GGUFTensorDescriptor(name="t", shape=(32,), ggml_type=GGML_TYPE_Q8_0, offset=0)
        assert desc.ggml_type == GGML_TYPE_Q8_0

    def test_descriptor_rejects_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="unsupported GGML type"):
            GGUFTensorDescriptor(name="t", shape=(32,), ggml_type=1, offset=0)


# ===================================================================
# Q8_0 byte size calculation (via descriptors)
# ===================================================================


class TestQ80ByteSize:
    def _byte_size(self, *shape: int) -> int:
        descs = build_gguf_tensor_descriptors(
            [_entry("t", *shape)], alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        last = descs[-1]
        return last.offset + (math.prod(shape) // QK8_0) * BLOCK_Q8_0_SIZE

    def test_one_block(self) -> None:
        assert self._byte_size(32) == 34

    def test_two_blocks(self) -> None:
        assert self._byte_size(64) == 68

    def test_2d_tensor(self) -> None:
        assert self._byte_size(4, 32) == 4 * 34

    def test_3d_tensor(self) -> None:
        assert self._byte_size(2, 4, 32) == 8 * 34

    def test_f32_byte_size_unchanged(self) -> None:
        build_gguf_tensor_descriptors([_entry("t", 64)], alignment=32)
        assert math.prod((64,)) * 4 == 256

    def test_single_element(self) -> None:
        with pytest.raises(ValueError, match="multiple of 32"):
            build_gguf_tensor_descriptors([_entry("t", 1)], alignment=32, ggml_type=GGML_TYPE_Q8_0)

    def test_31_elements_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of 32"):
            build_gguf_tensor_descriptors([_entry("t", 31)], alignment=32, ggml_type=GGML_TYPE_Q8_0)


# ===================================================================
# Q8_0 descriptor offsets
# ===================================================================


class TestQ80DescriptorOffsets:
    def test_first_offset_zero(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 32)], alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        assert descs[0].offset == 0

    def test_second_offset_follows_first(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 32), _entry("b", 32)],
            alignment=32,
            ggml_type=GGML_TYPE_Q8_0,
        )
        expected = ((34 + 31) // 32) * 32
        assert descs[1].offset == expected

    def test_varied_sizes(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 32), _entry("b", 64), _entry("c", 96)],
            alignment=32,
            ggml_type=GGML_TYPE_Q8_0,
        )
        assert descs[0].offset == 0
        off_b = ((34 + 31) // 32) * 32
        assert descs[1].offset == off_b
        off_c = ((off_b + 68 + 31) // 32) * 32
        assert descs[2].offset == off_c

    def test_sorted_by_name(self) -> None:
        tensors = [_entry("z", 32), _entry("a", 32), _entry("m", 32)]
        descs = build_gguf_tensor_descriptors(tensors, alignment=32, ggml_type=GGML_TYPE_Q8_0)
        assert [d.name for d in descs] == ["a", "m", "z"]

    def test_input_order_does_not_affect(self) -> None:
        t1 = [_entry("b", 32), _entry("a", 32)]
        t2 = [_entry("a", 32), _entry("b", 32)]
        r1 = build_gguf_tensor_descriptors(t1, alignment=32, ggml_type=GGML_TYPE_Q8_0)
        r2 = build_gguf_tensor_descriptors(t2, alignment=32, ggml_type=GGML_TYPE_Q8_0)
        assert r1 == r2

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate tensor names"):
            build_gguf_tensor_descriptors(
                [_entry("a", 32), _entry("a", 32)],
                alignment=32,
                ggml_type=GGML_TYPE_Q8_0,
            )

    def test_alignment_64(self) -> None:
        descs = build_gguf_tensor_descriptors(
            [_entry("a", 32), _entry("b", 32)],
            alignment=64,
            ggml_type=GGML_TYPE_Q8_0,
        )
        assert descs[0].offset == 0
        assert descs[1].offset == 64


# ===================================================================
# Payload building
# ===================================================================


class TestQ80PayloadBuild:
    def _prefix_size(self) -> int:
        tensors = _f32_tensors()
        normalized = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in sorted(tensors.items())
        }
        inventory = tuple(
            GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
            for name, tensor in normalized.items()
        )
        descriptors = build_gguf_tensor_descriptors(
            inventory, alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        return len(build_gguf_header_and_descriptors(_preflight(), descriptors))

    def test_payload_is_deterministic(self) -> None:
        first = build_gguf_q8_0_payload(_preflight(), _f32_tensors())
        second = build_gguf_q8_0_payload(
            _preflight(), dict(reversed(tuple(_f32_tensors().items())))
        )
        assert first == second

    def test_payload_matches_descriptor_offsets(self) -> None:
        payload = build_gguf_q8_0_payload(_preflight(), _f32_tensors())
        tensors = _f32_tensors()
        normalized = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in sorted(tensors.items())
        }
        inventory = tuple(
            GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
            for name, tensor in normalized.items()
        )
        descriptors = build_gguf_tensor_descriptors(
            inventory, alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        prefix_size = len(build_gguf_header_and_descriptors(_preflight(), descriptors))
        for desc in descriptors:
            block_bytes = (math.prod(desc.shape) // QK8_0) * BLOCK_Q8_0_SIZE
            seg = payload[prefix_size + desc.offset : prefix_size + desc.offset + block_bytes]
            assert len(seg) == block_bytes

    def test_payload_has_v3_header(self) -> None:
        payload = build_gguf_q8_0_payload(
            _preflight(tensor_count=1), {"t": torch.zeros(32, dtype=torch.float32)}
        )
        assert payload[:4] == b"GGUF"
        assert struct.unpack_from("<I", payload, 4)[0] == 3

    def test_tensor_count_must_match_preflight(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            build_gguf_q8_0_payload(_preflight(tensor_count=1), _f32_tensors())

    def test_non_f32_rejected(self) -> None:
        with pytest.raises(ValueError, match="torch.float32"):
            build_gguf_q8_0_payload(
                _preflight(tensor_count=1),
                {"t": torch.tensor([1.0], dtype=torch.float64)},
            )

    def test_non_multiple_of_32_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of 32"):
            build_gguf_q8_0_payload(
                _preflight(tensor_count=1),
                {"t": torch.tensor([1.0] * 31, dtype=torch.float32)},
            )

    def test_empty_tensors_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_gguf_q8_0_payload(_preflight(tensor_count=0), {})

    def test_tensor_inputs_are_detached(self) -> None:
        source = torch.tensor(
            [float(i) for i in range(32)], dtype=torch.float32, requires_grad=True
        )
        payload = build_gguf_q8_0_payload(_preflight(tensor_count=1), {"weight": source})
        assert source.requires_grad is True
        assert len(payload) > 0


# ===================================================================
# Writer (file-level)
# ===================================================================


class TestQ80Write:
    def test_write_creates_complete_file(self, tmp_path: Path) -> None:
        output = tmp_path / "bharat.gguf"
        result = write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), output)
        assert output.exists()
        assert output.read_bytes() == build_gguf_q8_0_payload(_preflight(), _f32_tensors())
        assert result.output_path == output
        assert result.bytes_written == output.stat().st_size
        assert result.payload_complete is True

    def test_existing_output_not_replaced(self, tmp_path: Path) -> None:
        output = tmp_path / "bharat.gguf"
        output.write_bytes(b"existing")
        with pytest.raises(FileExistsError, match="already exists"):
            write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), output)
        assert output.read_bytes() == b"existing"

    def test_wrong_suffix_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "model.bin"
        with pytest.raises(ValueError, match="must end with .gguf"):
            write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), output)

    def test_missing_parent_rejected(self, tmp_path: Path) -> None:
        output = tmp_path / "nonexistent" / "model.gguf"
        with pytest.raises(ValueError, match="output parent"):
            write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), output)


# ===================================================================
# Reader integration
# ===================================================================


class TestQ80ReaderIntegration:
    def test_reader_reads_q8_0_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), path)
        result = read_gguf_subset(path)
        assert result.version == 3
        assert result.tensors[0].ggml_type == GGML_TYPE_Q8_0
        assert result.tensors[1].ggml_type == GGML_TYPE_Q8_0

    def test_reader_reports_correct_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), path)
        result = read_gguf_subset(path)
        assert dict(result.metadata) == {
            "general.name": "Bharat",
            "general.scale": 1.0,
        }

    def test_reader_reports_correct_tensor_names(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), path)
        result = read_gguf_subset(path)
        assert [t.name for t in result.tensors] == ["a.weight", "z.weight"]

    def test_reader_reports_correct_offsets(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), path)
        result = read_gguf_subset(path)
        assert result.tensors[0].offset == 0
        assert result.tensors[1].offset > 0

    def test_reader_accepts_q8_0_type(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(
            _preflight(tensor_count=1), {"t": torch.zeros(32, dtype=torch.float32)}, path
        )
        result = read_gguf_subset(path)
        assert result.tensors[0].ggml_type == GGML_TYPE_Q8_0

    def test_reader_validates_file_bounds(self, tmp_path: Path) -> None:
        path = tmp_path / "bharat.gguf"
        write_gguf_q8_0_tensors(
            _preflight(tensor_count=1),
            {"t": torch.zeros(32, dtype=torch.float32)},
            path,
        )
        truncated = path.read_bytes()[:-1]
        path.write_bytes(truncated)
        with pytest.raises(ValueError, match="file bounds"):
            read_gguf_subset(path)

    def test_reader_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="regular file"):
            read_gguf_subset(tmp_path / "missing.gguf")

    def test_reader_rejects_wrong_suffix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="end with .gguf"):
            read_gguf_subset(tmp_path / "model.bin")

    def test_reader_rejects_invalid_magic(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors(
            [_entry("t", 32)], alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        header = build_gguf_header_and_descriptors(pf, descs)
        path.write_bytes(b"NOPE" + header[4:])
        with pytest.raises(ValueError, match="magic"):
            read_gguf_subset(path)


# ===================================================================
# Round-trip: write Q8_0, read, dequantize
# ===================================================================


class TestQ80RoundTrip:
    def _round_trip_values(self, tensors: dict[str, torch.Tensor]) -> dict[str, list[float]]:
        pf = _preflight(tensor_count=len(tensors))
        payload = build_gguf_q8_0_payload(pf, tensors)
        normalized = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in sorted(tensors.items())
        }
        inventory = tuple(
            GGUFTensorInventoryEntry(name=name, shape=tuple(tensor.shape))
            for name, tensor in normalized.items()
        )
        descriptors = build_gguf_tensor_descriptors(
            inventory, alignment=pf.alignment, ggml_type=GGML_TYPE_Q8_0
        )
        prefix = build_gguf_header_and_descriptors(pf, descriptors)
        prefix_size = len(prefix)

        result: dict[str, list[float]] = {}
        for desc in descriptors:
            element_count = math.prod(desc.shape)
            block_bytes = (element_count // QK8_0) * BLOCK_Q8_0_SIZE
            raw = payload[prefix_size + desc.offset : prefix_size + desc.offset + block_bytes]
            result[desc.name] = dequantize_q8_0(raw, element_count)
        return result

    def test_linear_ramp(self) -> None:
        data = [float(i) for i in range(32)]
        tensors = {"t": torch.tensor([data], dtype=torch.float32)}
        result = self._round_trip_values(tensors)
        for original, reconstructed in zip(data, result["t"], strict=True):
            assert reconstructed == pytest.approx(original, abs=0.15)

    def test_zeros(self) -> None:
        tensors = {"t": torch.zeros(64, dtype=torch.float32)}
        result = self._round_trip_values(tensors)
        assert all(v == 0.0 for v in result["t"])

    def test_negative_values(self) -> None:
        data = [float(-i) for i in range(32)]
        tensors = {"t": torch.tensor([data], dtype=torch.float32)}
        result = self._round_trip_values(tensors)
        for original, reconstructed in zip(data, result["t"], strict=True):
            assert reconstructed == pytest.approx(original, abs=0.15)

    def test_mixed_sign_values(self) -> None:
        data = [float(i - 15) for i in range(32)]
        tensors = {"t": torch.tensor([data], dtype=torch.float32)}
        result = self._round_trip_values(tensors)
        for original, reconstructed in zip(data, result["t"], strict=True):
            assert reconstructed == pytest.approx(original, abs=0.15)

    def test_multiple_tensors(self) -> None:
        tensors = {
            "a": torch.tensor([[float(i) for i in range(32)]], dtype=torch.float32),
            "b": torch.tensor([[float(i * 2) for i in range(32)]], dtype=torch.float32),
        }
        result = self._round_trip_values(tensors)
        assert len(result["a"]) == 32
        assert len(result["b"]) == 32

    def test_two_dimensional(self) -> None:
        data = [[float(i + j * 10) for i in range(32)] for j in range(4)]
        flat = [v for row in data for v in row]
        tensors = {"t": torch.tensor(data, dtype=torch.float32)}
        result = self._round_trip_values(tensors)
        max_abs = max(abs(v) for v in flat)
        expected_error = max_abs / 254.0
        tolerance = max(expected_error + 0.01, 0.01)
        for original, reconstructed in zip(flat, result["t"], strict=True):
            assert reconstructed == pytest.approx(original, abs=tolerance)


# ===================================================================
# Backward compatibility
# ===================================================================


class TestBackwardCompatibility:
    def test_f32_payload_still_works(self) -> None:
        tensors = {
            "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "b": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        }
        payload = build_gguf_f32_payload(_preflight(tensor_count=2), tensors)
        assert len(payload) > 0
        assert payload[:4] == b"GGUF"

    def test_f32_descriptors_unchanged(self) -> None:
        descs = build_gguf_tensor_descriptors([_entry("t", 64)], alignment=32)
        assert descs[0].ggml_type == GGML_TYPE_F32
        assert descs[0].offset == 0

    def test_reader_still_reads_f32(self, tmp_path: Path) -> None:
        path = tmp_path / "f32.gguf"
        pf = _preflight(tensor_count=2)
        payload = build_gguf_f32_payload(pf, _f32_tensors())
        path.write_bytes(payload)
        result = read_gguf_subset(path)
        assert all(t.ggml_type == GGML_TYPE_F32 for t in result.tensors)
        assert len(result.tensors) == 2

    def test_descriptor_only_output_unchanged(self, tmp_path: Path) -> None:
        output = tmp_path / "desc.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors([_entry("t", 4)], alignment=32)
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.descriptor_only is True
        assert result.bytes_written == output.stat().st_size

    def test_q8_0_tensors_in_descriptor_only_output(self, tmp_path: Path) -> None:
        output = tmp_path / "desc.gguf"
        pf = _preflight(tensor_count=1)
        descs = build_gguf_tensor_descriptors(
            [_entry("t", 32)], alignment=32, ggml_type=GGML_TYPE_Q8_0
        )
        result = write_gguf_header_and_descriptors(pf, descs, output)
        assert result.descriptor_count == 1
        assert result.tensor_count == 1


# ===================================================================
# Edge cases
# ===================================================================


class TestQ80EdgeCases:
    def test_exactly_one_block(self) -> None:
        tensors = {"t": torch.zeros(32, dtype=torch.float32)}
        payload = build_gguf_q8_0_payload(_preflight(tensor_count=1), tensors)
        assert len(payload) > 0

    def test_large_tensor(self) -> None:
        tensors = {"t": torch.zeros(4096, dtype=torch.float32)}
        payload = build_gguf_q8_0_payload(_preflight(tensor_count=1), tensors)
        assert len(payload) > 0

    def test_64_kb_tensor(self) -> None:
        tensors = {"t": torch.zeros(16384, dtype=torch.float32)}
        payload = build_gguf_q8_0_payload(_preflight(tensor_count=1), tensors)
        expected_bytes = (16384 // 32) * 34
        assert len(payload) > expected_bytes

    def test_sparse_tensor_rejected(self) -> None:
        sparse = torch.sparse_coo_tensor(indices=[[0]], values=[1.0], size=[32])
        with pytest.raises(ValueError, match="dense strided"):
            build_gguf_q8_0_payload(_preflight(tensor_count=1), {"t": sparse})

    def test_non_contiguous_tensor(self) -> None:
        source = torch.tensor([[float(i) for i in range(32)]], dtype=torch.float32)
        view = source.t()
        payload = build_gguf_q8_0_payload(_preflight(tensor_count=1), {"t": view})
        assert len(payload) > 0

    def test_result_counts(self, tmp_path: Path) -> None:
        output = tmp_path / "bharat.gguf"
        result = write_gguf_q8_0_tensors(_preflight(), _f32_tensors(), output)
        assert result.tensor_count == 2
        assert result.metadata_count == 2
        assert result.tensor_data_bytes > 0
        assert result.alignment == 32
