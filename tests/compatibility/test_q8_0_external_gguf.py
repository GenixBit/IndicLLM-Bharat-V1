from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_export_plan.py"

Q8_0 = gguf.GGMLQuantizationType.Q8_0
F32_T = gguf.GGMLQuantizationType.F32

QK8_0 = 32
BLOCK_Q8_0_SIZE = 34


def _make_fixture(tmp_path: Path) -> Path:
    import torch

    cp = tmp_path / "checkpoint"
    cp.mkdir(parents=True, exist_ok=True)
    t1 = [float(i + 1) * (-1 if i % 2 else 1) for i in range(32)]
    t2 = [0.0] * 32 + [float(i + 1) for i in range(32)]
    t3 = [float(i % 7 + 1) * (-1 if (i // 7) % 2 else 1) for i in range(96)]
    t4 = [float(i * 2 + 1) * 0.5 * (-1 if i % 3 == 0 else 1) for i in range(32)]
    state = {
        "tensor_32": torch.tensor(t1, dtype=torch.float32).reshape(32),
        "tensor_64": torch.tensor(t2, dtype=torch.float32).reshape(64),
        "tensor_96": torch.tensor(t3, dtype=torch.float32).reshape(96),
        "tensor_2d": torch.tensor(t4, dtype=torch.float32).reshape(1, 32),
    }
    torch.save(state, cp / "model.pt")
    (cp / "model.gguf").write_bytes(b"placeholder")
    return cp


def _make_metadata(path: Path, tensor_count: int) -> None:
    meta = {
        "schema_version": 1,
        "architecture": "compatibility-fixture",
        "alignment": 32,
        "tensor_count": tensor_count,
        "output_file": "model.gguf",
        "metadata": [{"key": "general.name", "value_type": "string", "value": "compat"}],
    }
    path.write_text(json.dumps(meta))


def _export(tmp_path: Path, tensor_type: str, suffix: str = "") -> dict:
    cp = _make_fixture(tmp_path)
    meta = tmp_path / "meta.json"
    _make_metadata(meta, 4)
    out_dir = tmp_path / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"model{suffix}_{tensor_type}.gguf"
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--checkpoint-path",
            str(cp),
            "--output-path",
            str(out),
            "--format",
            "gguf",
            "--model-name",
            "compat-fixture",
            "--gguf-metadata-path",
            str(meta),
            "--gguf-tensor-type",
            tensor_type,
            "--execute",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    parsed = json.loads(result.stdout)
    parsed["_output_path"] = str(out)
    return parsed


def _read_gguf(path: str | Path) -> gguf.GGUFReader:
    return gguf.GGUFReader(path)


def _source_values() -> dict[str, np.ndarray]:
    t1 = np.array([float(i + 1) * (-1 if i % 2 else 1) for i in range(32)], dtype=np.float32)
    t2 = np.array([0.0] * 32 + [float(i + 1) for i in range(32)], dtype=np.float32)
    t3 = np.array(
        [float(i % 7 + 1) * (-1 if (i // 7) % 2 else 1) for i in range(96)], dtype=np.float32
    )
    t4 = np.array(
        [float(i * 2 + 1) * 0.5 * (-1 if i % 3 == 0 else 1) for i in range(32)], dtype=np.float32
    )
    return {"tensor_32": t1, "tensor_64": t2, "tensor_96": t3, "tensor_2d": t4}


pytestmark = [pytest.mark.compatibility]


class TestExternalGgufCompat:
    def test_cli_export_succeeds(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        assert data["gguf_tensor_type"] == "q8_0"
        out = Path(data["_output_path"])
        assert out.exists()
        assert out.stat().st_size > 0
        assert data["bytes_written"] == out.stat().st_size

    def test_independent_parses(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        assert len(reader.tensors) == 4

    def test_tensor_count_matches(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        assert len(reader.tensors) == 4

    def test_tensor_names_match(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        names = sorted(t.name for t in reader.tensors)
        assert names == sorted(["tensor_32", "tensor_64", "tensor_96", "tensor_2d"])

    def test_tensor_shapes_match(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        shapes = {t.name: list(t.shape) for t in reader.tensors}
        assert shapes["tensor_32"] == [32]
        assert shapes["tensor_64"] == [64]
        assert shapes["tensor_96"] == [96]
        assert shapes["tensor_2d"] == [32, 1]

    def test_all_tensors_report_q8_0(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        for t in reader.tensors:
            assert t.tensor_type == Q8_0, f"{t.name} has type {t.tensor_type}"

    def test_payload_sizes_accepted(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        for t in reader.tensors:
            assert t.n_bytes > 0
            assert t.data_offset > 0

    def test_alignment_accepted(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        assert reader.alignment > 0
        assert (reader.alignment & (reader.alignment - 1)) == 0

    def test_dequantization_accuracy(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        source = _source_values()
        for t in reader.tensors:
            deq = gguf.dequantize(t.data, Q8_0).flatten()
            src = source[t.name].flatten()
            assert deq.shape == src.shape, f"{t.name}: shape {deq.shape} vs {src.shape}"
            err = np.abs(deq - src)
            max_err = float(err.max())
            rmse = float(np.sqrt(np.mean(err**2)))
            src_norm = float(np.linalg.norm(src))
            if src_norm > 0:
                deq_norm = float(np.linalg.norm(deq))
                cos_sim = float(np.dot(deq, src) / (deq_norm * src_norm))
            else:
                cos_sim = 1.0
            assert max_err < 1.0, f"{t.name}: max_err={max_err}"
            assert rmse < 0.5, f"{t.name}: rmse={rmse}"
            assert cos_sim > 0.99, f"{t.name}: cos_sim={cos_sim}"

    def test_zero_block_reconstructs(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        tensors_by_name = {t.name: t for t in reader.tensors}
        t = tensors_by_name["tensor_64"]
        deq = gguf.dequantize(t.data, Q8_0).flatten()
        first_block = deq[:32]
        assert np.allclose(first_block, 0.0, atol=1e-6)

    def test_multi_block_reconstructs(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        tensors_by_name = {t.name: t for t in reader.tensors}
        t = tensors_by_name["tensor_96"]
        deq = gguf.dequantize(t.data, Q8_0).flatten()
        assert len(deq) == 96
        source = _source_values()
        src = source["tensor_96"].flatten()
        for i in range(3):
            block = deq[i * QK8_0 : (i + 1) * QK8_0]
            src_block = src[i * QK8_0 : (i + 1) * QK8_0]
            err = np.abs(block - src_block)
            assert float(err.mean()) < 0.5, f"block {i}: mean_err={float(err.mean())}"

    def test_byte_level_block(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        out = Path(data["_output_path"])
        raw = out.read_bytes()
        reader = _read_gguf(str(out))
        tensors_by_name = {t.name: t for t in reader.tensors}
        t = tensors_by_name["tensor_32"]
        block_offs = t.data_offset
        block_bytes = raw[block_offs : block_offs + BLOCK_Q8_0_SIZE]
        assert len(block_bytes) == BLOCK_Q8_0_SIZE
        scale_bits = block_bytes[0] | (block_bytes[1] << 8)
        scale = struct.unpack("<e", struct.pack("<H", scale_bits))[0]
        assert scale > 0
        quants_raw = list(block_bytes[2:])
        assert len(quants_raw) == QK8_0
        quants_signed = [v - 256 if v >= 128 else v for v in quants_raw]
        for j in range(QK8_0):
            expected = quants_signed[j] * scale
            deq_all = gguf.dequantize(t.data, Q8_0).flatten()
            assert deq_all[j] == pytest.approx(expected, rel=1e-5)

    def test_f32_control_parses(self, tmp_path: Path) -> None:
        data_f32 = _export(tmp_path, "f32")
        reader = _read_gguf(data_f32["_output_path"])
        assert len(reader.tensors) == 4
        for t in reader.tensors:
            assert t.tensor_type == F32_T, f"{t.name} has type {t.tensor_type}"

    def test_q8_0_smaller_than_f32(self, tmp_path: Path) -> None:
        data_f32 = _export(tmp_path, "f32")
        data_q8 = _export(tmp_path, "q8_0")
        f32_size = Path(data_f32["_output_path"]).stat().st_size
        q8_size = Path(data_q8["_output_path"]).stat().st_size
        assert q8_size < f32_size

    def test_repeated_exports_identical(self, tmp_path: Path) -> None:
        r1 = _export(tmp_path, "q8_0", suffix="_r1")
        r2 = _export(tmp_path, "q8_0", suffix="_r2")
        p1 = Path(r1["_output_path"]).read_bytes()
        p2 = Path(r2["_output_path"]).read_bytes()
        assert p1 == p2

    def test_truncated_file_rejected(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        out = Path(data["_output_path"])
        truncated = out.with_suffix(".trunc.gguf")
        truncated.write_bytes(out.read_bytes()[:-BLOCK_Q8_0_SIZE])
        with pytest.raises((ValueError, OSError, Exception)):
            _read_gguf(truncated)

    def test_corrupted_tensor_data_changes_dequant(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        out = Path(data["_output_path"])
        reader = _read_gguf(str(out))
        data_offs = reader.tensors[0].data_offset
        raw = bytearray(out.read_bytes())
        raw[data_offs + 10] ^= 0xFF
        bad = out.with_suffix(".bad_data.gguf")
        bad.write_bytes(bytes(raw))
        reader2 = _read_gguf(str(bad))
        deq_orig = gguf.dequantize(reader.tensors[0].data, Q8_0)
        deq_bad = gguf.dequantize(reader2.tensors[0].data, Q8_0)
        assert not np.allclose(deq_orig, deq_bad)

    def test_corrupted_type_rejected(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        out = Path(data["_output_path"])
        reader = _read_gguf(str(out))
        raw_dtype = reader.tensors[0].field.parts[4]
        dtype_file_offset = raw_dtype.ctypes.data - reader.data.ctypes.data
        raw = bytearray(out.read_bytes())
        assert raw[dtype_file_offset] == 8
        raw[dtype_file_offset] = 99
        bad = out.with_suffix(".bad_type.gguf")
        bad.write_bytes(bytes(raw))
        with pytest.raises((ValueError, OSError, Exception)):
            _read_gguf(bad)

    def test_malformed_magic_rejected(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        out = Path(data["_output_path"])
        raw = bytearray(out.read_bytes())
        raw[0:4] = b"BADD"
        bad = out.with_suffix(".bad_magic.gguf")
        bad.write_bytes(bytes(raw))
        with pytest.raises((ValueError, OSError, Exception)):
            _read_gguf(bad)

    def test_no_repo_reader_used(self) -> None:
        assert "bharat" not in Path(gguf.__file__).parts

    def test_no_repo_dequantizer_used(self, tmp_path: Path) -> None:
        data = _export(tmp_path, "q8_0")
        reader = _read_gguf(data["_output_path"])
        for t in reader.tensors:
            deq = gguf.dequantize(t.data, Q8_0)
            assert deq.dtype == np.float32
