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
# 4. --execute with GGUF is rejected
# ---------------------------------------------------------------------------


class TestExecuteGGUF:
    def test_execute_gguf_rejected(self, tmp_path: Path) -> None:
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
                "--execute",
            ]
        )
        assert result.returncode != 0
        assert "real GGUF export is not implemented" in result.stderr

    def test_execute_gguf_creates_no_file(self, tmp_path: Path) -> None:
        cp = _bharat_checkpoint(tmp_path)
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
                "--execute",
            ]
        )
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
