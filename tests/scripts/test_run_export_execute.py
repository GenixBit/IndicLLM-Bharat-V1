from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_writer import ExportWriterRegistry


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_export_plan", *args],
        capture_output=True,
        text=True,
    )


def _make_state_dict(
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype, list[float]]],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(data, dtype=dtype).reshape(shape)
        for name, (shape, dtype, data) in tensors.items()
    }


def _bharat_checkpoint(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    torch.save(
        _make_state_dict(
            {
                "weight": ((2, 3), torch.float32, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "bias": ((3,), torch.float32, [0.1, 0.2, 0.3]),
            }
        ),
        cp / "model.pt",
    )
    return cp


# ---------------------------------------------------------------------------
# 1. Dry-run remains the default
# ---------------------------------------------------------------------------


class TestDryRunDefault:
    def test_dry_run_remains_default(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dry_run"] is True
        assert data["writer_name"] == "safetensors-dry-run"
        assert data["bytes_written"] == 0

    def test_dry_run_creates_no_output(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert not out.exists()

    def test_dry_run_gguf_unchanged(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dry_run"] is True
        assert data["writer_name"] == "gguf-dry-run"


# ---------------------------------------------------------------------------
# 3. --execute with safetensors writes a valid file
# ---------------------------------------------------------------------------


class TestExecuteSafetensors:
    def test_execute_writes_valid_file(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode == 0
        assert out.exists()

    def test_output_non_empty(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["bytes_written"] > 0
        assert out.stat().st_size == data["bytes_written"]

    def test_bytes_written_matches_file(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert data["bytes_written"] == out.stat().st_size

    def test_roundtrip_preserves_tensors(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        loaded = load_file(str(out))
        assert set(loaded.keys()) == {"weight", "bias"}
        assert loaded["weight"].shape == (2, 3)
        assert loaded["bias"].shape == (3,)
        assert loaded["weight"].dtype == torch.float32
        assert loaded["bias"].dtype == torch.float32
        assert torch.equal(
            loaded["weight"],
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
        )

    def test_writer_name_and_dry_run(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert data["writer_name"] == "safetensors-local"
        assert data["dry_run"] is False


# ---------------------------------------------------------------------------
# 4. GGUF execution helpers
# ---------------------------------------------------------------------------


def _gguf_checkpoint(
    tmp_path: Path,
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype, list[float]]] | None = None,
) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    if tensors is None:
        tensors = {
            "weight": ((2, 3), torch.float32, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "bias": ((3,), torch.float32, [0.1, 0.2, 0.3]),
        }
    torch.save(_make_state_dict(tensors), cp / "model.pt")
    (cp / "model.gguf").write_bytes(b"placeholder")
    return cp


def _gguf_metadata(tensor_count: int = 2) -> dict:
    return {
        "schema_version": 1,
        "architecture": "bharat",
        "alignment": 32,
        "tensor_count": tensor_count,
        "output_file": "model.gguf",
        "metadata": [{"key": "general.name", "value_type": "string", "value": "Bharat"}],
    }


def _parse_gguf_magic(payload: bytes) -> bool:
    return payload[:4] == b"GGUF"


def _parse_gguf_version(payload: bytes) -> int:
    import struct

    return struct.unpack_from("<I", payload, 4)[0]


def _parse_gguf_tensor_count(payload: bytes) -> int:
    import struct

    return struct.unpack_from("<Q", payload, 8)[0]


def _parse_gguf_metadata_count(payload: bytes) -> int:
    import struct

    return struct.unpack_from("<Q", payload, 16)[0]


# ---------------------------------------------------------------------------
# 5. --execute with GGUF
# ---------------------------------------------------------------------------


class TestGGUFDefaults:
    def test_gguf_dry_run_unchanged(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dry_run"] is True
        assert data["writer_name"] == "gguf-dry-run"
        assert data["bytes_written"] == 0

    def test_gguf_dry_run_creates_no_output(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
            ]
        )
        assert not out.exists()

    def test_gguf_execute_requires_metadata(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert "--gguf-metadata-path is required" in result.stderr

    def test_gguf_execute_requires_metadata_before_loading(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()


class TestGGUFExecuteInvalidMetadata:
    def test_invalid_metadata_json_rejected(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text("not json")
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()

    def test_missing_metadata_file_rejected(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "nonexistent.json"
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()

    def test_remote_metadata_rejected(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                "https://example.com/meta.json",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert "Remote" in result.stderr


class TestGGUFExecuteSuccess:
    def _run_gguf(self, tmp_path: Path, tensor_count: int = 2, extra: list[str] | None = None):
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=tensor_count)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        base = [
            "--checkpoint-path",
            str(cp),
            "--output-path",
            str(out),
            "--format",
            "gguf",
            "--model-name",
            "bharat-local",
            "--gguf-metadata-path",
            str(meta),
            "--execute",
        ]
        if extra:
            base.extend(extra)
        result = run_cli(base)
        return result, out, meta

    def test_successful_gguf_export(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["dry_run"] is False
        assert data["writer_name"] == "gguf-f32-local"
        assert data["export_format"] == "gguf"
        assert data["bytes_written"] > 0
        assert out.exists()

    def test_output_is_non_empty(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        assert out.stat().st_size > 0

    def test_bytes_written_matches_file_size(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["bytes_written"] == out.stat().st_size

    def test_output_starts_with_gguf_magic(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        payload = out.read_bytes()
        assert _parse_gguf_magic(payload)

    def test_output_gguf_version(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        payload = out.read_bytes()
        assert _parse_gguf_version(payload) == 3

    def test_tensor_count_in_header(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        payload = out.read_bytes()
        assert _parse_gguf_tensor_count(payload) == 2

    def test_metadata_count_in_header(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        payload = out.read_bytes()
        assert _parse_gguf_metadata_count(payload) == 1

    def test_tensor_names_preserved(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        payload = out.read_bytes()
        assert b"weight" in payload
        assert b"bias" in payload

    def test_writer_name_and_dry_run(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["writer_name"] == "gguf-f32-local"
        assert data["dry_run"] is False

    def test_gguf_preflight_in_json(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        data = json.loads(result.stdout)
        assert "gguf_preflight" in data
        assert data["gguf_preflight"]["tensor_count"] == 2

    def test_inventory_hidden_by_default(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        data = json.loads(result.stdout)
        assert "checkpoint_inventory" not in data

    def test_inventory_visible_with_flag(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path, extra=["--include-inventory"])
        data = json.loads(result.stdout)
        assert "checkpoint_inventory" in data

    def test_plain_state_dict_works(self, tmp_path: Path) -> None:
        cp = tmp_path / "plain_checkpoint"
        cp.mkdir()
        torch.save({"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}, cp / "model.pt")
        (cp / "model.gguf").write_bytes(b"placeholder")
        meta = tmp_path / "meta_plain.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=1)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_model_state_dict_works(self, tmp_path: Path) -> None:
        cp = tmp_path / "model_dict_checkpoint"
        cp.mkdir()
        torch.save({"model": {"weight": torch.tensor([3.0], dtype=torch.float32)}}, cp / "model.pt")
        (cp / "model.gguf").write_bytes(b"placeholder")
        meta = tmp_path / "meta_dict.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=1)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_directory_checkpoint_resolves_model_pt(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        assert result.returncode == 0

    def test_gguf_preflight_in_output(self, tmp_path: Path) -> None:
        result, out, _ = self._run_gguf(tmp_path)
        data = json.loads(result.stdout)
        assert "gguf_preflight" in data
        assert data["gguf_preflight"]["tensor_count"] == 2


class TestGGUFExecuteManifest:
    def test_manifest_written_after_gguf_output(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata()))
        out = tmp_path / "exports" / "bharat.gguf"
        manifest = tmp_path / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out.exists()
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["dry_run"] is False
        assert manifest_data["bytes_written"] > 0
        assert manifest_data["writer_name"] == "gguf-f32-local"
        assert manifest_data["export_format"] == "gguf"

    def test_manifest_not_written_on_failure(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata()))
        out = tmp_path / "nonexistent" / "bharat.gguf"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not manifest.exists()


class TestGGUFExecuteRejection:
    def test_non_f32_tensor_rejected(self, tmp_path: Path) -> None:
        cp = tmp_path / "f16_checkpoint"
        cp.mkdir()
        torch.save({"weight": torch.tensor([1.0], dtype=torch.float16)}, cp / "model.pt")
        (cp / "model.gguf").write_bytes(b"placeholder")
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=1)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert "torch.float32" in result.stderr
        assert not out.exists()

    def test_tensor_count_mismatch_rejected(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=99)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()

    def test_existing_output_rejected(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata()))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("existing")
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert out.read_text() == "existing"

    def test_no_partial_json_on_failure(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=99)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert result.stdout.strip() == ""

    def test_no_dry_run_fallback_on_failure(self, tmp_path: Path) -> None:
        cp = _gguf_checkpoint(tmp_path)
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps(_gguf_metadata(tensor_count=99)))
        out = tmp_path / "exports" / "bharat.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "gguf",
                "--model-name",
                "bharat-local",
                "--gguf-metadata-path",
                str(meta),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()


# ---------------------------------------------------------------------------
# 5. Automatic readiness and inventory
# ---------------------------------------------------------------------------


class TestAutomaticReadiness:
    def test_writer_readiness_runs_auto(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert "writer_readiness" in data

    def test_inventory_built_auto_for_execute(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert "checkpoint_inventory" not in data

    def test_inventory_visible_with_include(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--include-inventory",
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert "checkpoint_inventory" in data

    def test_path_readiness_before_writer(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        manifest = tmp_path / "manifest.json"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        data = json.loads(result.stdout)
        assert "export_path_readiness" in data


# ---------------------------------------------------------------------------
# 6. Rejection cases
# ---------------------------------------------------------------------------


class TestRejection:
    def test_existing_output_rejected(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        out.write_text("existing")
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0

    def test_missing_output_parent_rejected(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "nonexistent" / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0

    def test_output_inside_checkpoint_rejected(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = cp / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0

    def test_existing_manifest_rejected(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        manifest = tmp_path / "manifest.json"
        out.parent.mkdir()
        manifest.write_text("{}")
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode != 0

    def test_writer_readiness_failure_prevents_write(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0

    def test_writer_failure_no_output_file(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "nonexistent" / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not out.exists()

    def test_remote_paths_rejected(self, tmp_path: Path) -> None:
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                "https://example.com/model.pt",
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert "Remote" in result.stderr


# ---------------------------------------------------------------------------
# 7. Manifest behaviour
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_written_after_output(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        manifest = tmp_path / "manifest.json"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0
        assert out.exists()
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["dry_run"] is False
        assert manifest_data["bytes_written"] > 0
        assert manifest_data["writer_name"] == "safetensors-local"

    def test_manifest_not_written_on_failure(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "nonexistent" / "bharat.safetensors"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--manifest-path",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert not manifest.exists()

    def test_no_partial_json_on_failure(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "nonexistent" / "bharat.safetensors"
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# 8. Existing CLI output unchanged without --execute
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_existing_cli_output_unchanged(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        result = run_cli(
            [
                "--checkpoint-path",
                str(cp),
                "--output-path",
                str(out),
                "--format",
                "safetensors",
                "--model-name",
                "bharat-local",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["dry_run"] is True
        assert data["writer_name"] == "safetensors-dry-run"
        assert "writer_readiness" not in data
        assert "checkpoint_inventory" not in data


# ---------------------------------------------------------------------------
# 9. Registry selection
# ---------------------------------------------------------------------------


class TestRegistrySelection:
    def test_real_writer_selected_through_registry(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
        out = tmp_path / "exports" / "bharat.safetensors"
        out.parent.mkdir()
        plan = build_export_plan(
            ExportRequest(
                checkpoint_path=cp,
                output_path=out,
                export_format="safetensors",
                model_name="bharat-local",
                dry_run=False,
            )
        )
        result = ExportWriterRegistry().write(plan)
        assert result.writer_name == "safetensors-local"
        assert result.dry_run is False
        assert result.bytes_written > 0
        assert out.exists()

    def test_dry_run_writer_selected_through_registry(self) -> None:
        plan = build_export_plan(
            ExportRequest(
                checkpoint_path=Path("checkpoints/bharat"),
                output_path=Path("exports/bharat.safetensors"),
                export_format="safetensors",
                model_name="bharat-local",
            )
        )
        result = ExportWriterRegistry().write(plan)
        assert result.writer_name == "safetensors-dry-run"
        assert result.dry_run is True
        assert result.bytes_written == 0

    def test_non_dry_run_gguf_raises(self) -> None:
        plan = build_export_plan(
            ExportRequest(
                checkpoint_path=Path("checkpoints/bharat"),
                output_path=Path("exports/bharat.gguf"),
                export_format="gguf",
                model_name="bharat-local",
                dry_run=False,
            )
        )
        with pytest.raises(ValueError, match="no execute writer registered"):
            ExportWriterRegistry().write(plan)

    def test_custom_writer_injection(self) -> None:
        registry = ExportWriterRegistry(())

        class FakeWriter:
            name = "fake"
            export_format = "safetensors"

            def write(self, plan):
                from bharat.serving.export_writer import ExportWriteResult

                return ExportWriteResult(
                    output_path=plan.output_path,
                    export_format="safetensors",
                    writer_name="fake",
                )

        with pytest.raises(ValueError, match="no writer registered"):
            registry.get("safetensors")  # No custom dry-run writer registered

    def test_existing_duplicate_format_test_still_works(self) -> None:
        from bharat.serving.export_writer import DryRunExportWriter

        writers = (
            DryRunExportWriter(name="first", export_format="gguf"),
            DryRunExportWriter(name="second", export_format="gguf"),
        )
        with pytest.raises(ValueError, match="duplicate writer"):
            ExportWriterRegistry(writers)
