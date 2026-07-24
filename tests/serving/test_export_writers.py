from __future__ import annotations

from pathlib import Path

import pytest

from bharat.serving.export import (
    _WRITER_REGISTRY,
    DryRunExportWriter,
    ExportPlan,
    ExportRequest,
    ExportResult,
    build_export_plan,
    get_writer,
    register_writer,
    run_export,
)


def test_dry_run_safetensors_success() -> None:
    plan = ExportPlan(
        checkpoint_path=Path("ckpt"),
        output_path=Path("out/model.safetensors"),
        export_format="safetensors",
        model_name="test-model",
        dry_run=True,
    )
    writer = DryRunExportWriter("safetensors")
    result = writer.write(plan)
    assert result.success
    assert "Dry-run" in result.message


def test_dry_run_gguf_success() -> None:
    plan = ExportPlan(
        checkpoint_path=Path("ckpt"),
        output_path=Path("out/model.gguf"),
        export_format="gguf",
        model_name="test-model",
        dry_run=True,
    )
    writer = DryRunExportWriter("gguf")
    result = writer.write(plan)
    assert result.success
    assert result.export_format == "gguf"


def test_dry_run_wrong_suffix_fails() -> None:
    plan = ExportPlan(
        checkpoint_path=Path("ckpt"),
        output_path=Path("out/model.bin"),
        export_format="safetensors",
        model_name="test-model",
        dry_run=True,
    )
    writer = DryRunExportWriter("safetensors")
    result = writer.write(plan)
    assert not result.success
    assert ".safetensors" in result.message


def test_export_result_to_dict() -> None:
    result = ExportResult(
        output_path=Path("out/model.safetensors"),
        export_format="safetensors",
        model_name="m",
        success=True,
        message="ok",
    )
    d = result.to_dict()
    assert d["success"] is True
    assert d["output_path"] == "out/model.safetensors"


def test_export_result_to_json() -> None:
    result = ExportResult(
        output_path=Path("out/model.gguf"),
        export_format="gguf",
        model_name="m",
        success=True,
        message="ok",
    )
    j = result.to_json()
    assert '"success": true' in j
    assert '"model_name": "m"' in j


def test_registry_get_writer_safetensors() -> None:
    writer = get_writer("safetensors")
    assert isinstance(writer, DryRunExportWriter)
    assert writer.export_format == "safetensors"


def test_registry_get_writer_gguf() -> None:
    writer = get_writer("gguf")
    assert isinstance(writer, DryRunExportWriter)
    assert writer.export_format == "gguf"


def test_registry_missing_format_raises() -> None:
    with pytest.raises(ValueError, match="No writer registered"):
        get_writer("ggml")  # type: ignore[arg-type]


def test_register_custom_writer() -> None:
    class FakeWriter:
        def __init__(self, export_format: str) -> None:
            self.export_format = export_format

        def write(self, plan: ExportPlan) -> ExportResult:
            return ExportResult(
                output_path=plan.output_path,
                export_format=plan.export_format,
                model_name=plan.model_name,
                success=True,
                message="fake",
            )

    register_writer("safetensors", FakeWriter)
    writer = get_writer("safetensors")
    assert isinstance(writer, FakeWriter)
    _WRITER_REGISTRY["safetensors"] = DryRunExportWriter


def test_run_export_with_plan() -> None:
    request = ExportRequest(
        checkpoint_path=Path("ckpt"),
        output_path=Path("out/model.safetensors"),
        export_format="safetensors",
        model_name="test-model",
    )
    plan = build_export_plan(request)
    result = run_export(plan)
    assert result.success
    assert result.model_name == "test-model"


def test_run_export_wrong_suffix() -> None:
    plan = ExportPlan(
        checkpoint_path=Path("ckpt"),
        output_path=Path("out/model.bin"),
        export_format="gguf",
        model_name="test-model",
        dry_run=True,
    )
    result = run_export(plan)
    assert not result.success
