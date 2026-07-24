from pathlib import Path

import pytest

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_writer import (
    DryRunExportWriter,
    ExportWriteResult,
    ExportWriterRegistry,
)


def _plan(export_format: str, suffix: str):
    return build_export_plan(
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path(f"exports/bharat{suffix}"),
            export_format=export_format,  # type: ignore[arg-type]
            model_name="bharat-local",
        )
    )


def test_registry_writes_safetensors_dry_run() -> None:
    result = ExportWriterRegistry().write(_plan("safetensors", ".safetensors"))

    assert result.writer_name == "safetensors-dry-run"
    assert result.dry_run is True
    assert result.bytes_written == 0
    assert result.to_dict()["output_path"] == "exports/bharat.safetensors"


def test_registry_writes_gguf_dry_run() -> None:
    result = ExportWriterRegistry().write(_plan("gguf", ".gguf"))

    assert result.export_format == "gguf"
    assert result.writer_name == "gguf-dry-run"


def test_writer_rejects_mismatched_format() -> None:
    writer = DryRunExportWriter(name="gguf-only", export_format="gguf")

    with pytest.raises(ValueError, match="does not support"):
        writer.write(_plan("safetensors", ".safetensors"))


def test_registry_rejects_duplicate_formats() -> None:
    writers = (
        DryRunExportWriter(name="first", export_format="gguf"),
        DryRunExportWriter(name="second", export_format="gguf"),
    )

    with pytest.raises(ValueError, match="duplicate writer"):
        ExportWriterRegistry(writers)


def test_dry_run_result_rejects_written_bytes() -> None:
    with pytest.raises(ValueError, match="zero bytes"):
        ExportWriteResult(
            output_path=Path("exports/bharat.gguf"),
            export_format="gguf",
            writer_name="invalid",
            dry_run=True,
            bytes_written=1,
        )
