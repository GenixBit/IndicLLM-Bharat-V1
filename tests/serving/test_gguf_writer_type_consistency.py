from pathlib import Path

import pytest

from bharat.serving.export import ExportPlan
from bharat.serving.export_writer import (
    ExportWriterRegistry,
    LocalGGUFF32ExportWriter,
    LocalGGUFQ8_0ExportWriter,
)
from bharat.serving.gguf_preflight import GGUFPreflightResult


def _preflight(tensor_type: str) -> GGUFPreflightResult:
    return GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=1,
        output_file="model.gguf",
        metadata=(),
        gguf_tensor_type=tensor_type,
    )


def _plan(tensor_type: str) -> ExportPlan:
    return ExportPlan(
        checkpoint_path=Path("checkpoint/model.pt"),
        output_path=Path("exports/model.gguf"),
        export_format="gguf",
        model_name="bharat-local",
        dry_run=False,
        gguf_tensor_type=tensor_type,
    )


def test_registry_rejects_preflight_selection_mismatch() -> None:
    with pytest.raises(ValueError, match="preflight tensor type does not match"):
        ExportWriterRegistry(
            gguf_preflight=_preflight("q8_0"),
            gguf_tensor_type="f32",
        )


def test_f32_writer_rejects_q8_0_plan_before_checkpoint_loading() -> None:
    writer = LocalGGUFF32ExportWriter(preflight=_preflight("f32"))

    with pytest.raises(ValueError, match="requires gguf_tensor_type 'f32'"):
        writer.write(_plan("q8_0"))


def test_q8_0_writer_rejects_f32_plan_before_checkpoint_loading() -> None:
    writer = LocalGGUFQ8_0ExportWriter(preflight=_preflight("q8_0"))

    with pytest.raises(ValueError, match="requires gguf_tensor_type 'q8_0'"):
        writer.write(_plan("f32"))


def test_writer_rejects_mismatched_preflight_before_checkpoint_loading() -> None:
    writer = LocalGGUFQ8_0ExportWriter(preflight=_preflight("f32"))

    with pytest.raises(ValueError, match="requires preflight gguf_tensor_type 'q8_0'"):
        writer.write(_plan("q8_0"))
