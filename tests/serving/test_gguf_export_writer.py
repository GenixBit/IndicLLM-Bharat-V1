from pathlib import Path

import pytest
import torch

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_writer import ExportWriterRegistry, LocalGGUFF32ExportWriter
from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult


def _preflight(*, tensor_count: int = 1) -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=tensor_count,
        output_file="model.pt",
        metadata=(GGUFMetadataEntry("general.name", "string", "Bharat"),),
    )


def _plan(checkpoint_path: Path, output_path: Path):
    return build_export_plan(
        ExportRequest(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            export_format="gguf",
            model_name="bharat-local",
            dry_run=False,
        )
    )


def test_registry_executes_local_gguf_f32_writer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.gguf"
    torch.save(
        {
            "model": {
                "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            },
        },
        checkpoint,
    )

    result = ExportWriterRegistry(gguf_preflight=_preflight()).write(
        _plan(checkpoint, output)
    )

    assert result.writer_name == "gguf-f32-local"
    assert result.export_format == "gguf"
    assert result.dry_run is False
    assert result.bytes_written == output.stat().st_size
    assert output.read_bytes().startswith(b"GGUF")


def test_registry_requires_preflight_for_gguf_execution(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "model.pt", tmp_path / "model.gguf")

    with pytest.raises(ValueError, match="no execute writer registered"):
        ExportWriterRegistry().write(plan)


def test_writer_rejects_non_f32_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save({"weight": torch.tensor([1.0], dtype=torch.float64)}, checkpoint)

    with pytest.raises(ValueError, match="torch.float32"):
        LocalGGUFF32ExportWriter(_preflight()).write(
            _plan(checkpoint, tmp_path / "model.gguf"),
        )


def test_writer_loads_checkpoint_directory_model_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save(
        {"weight": torch.tensor([1.0], dtype=torch.float32)},
        checkpoint_dir / "model.pt",
    )

    output = tmp_path / "model.gguf"
    result = LocalGGUFF32ExportWriter(_preflight()).write(
        _plan(checkpoint_dir, output)
    )

    assert result.bytes_written > 0
    assert output.is_file()
