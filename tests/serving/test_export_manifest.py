from pathlib import Path

import pytest

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_manifest import ExportManifest, write_export_manifest
from bharat.serving.export_writer import ExportWriteResult, ExportWriterRegistry


def _plan_and_result():
    plan = build_export_plan(
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path("exports/bharat.safetensors"),
            export_format="safetensors",
            model_name="bharat-local",
        )
    )
    return plan, ExportWriterRegistry().write(plan)


def test_manifest_is_deterministic() -> None:
    plan, result = _plan_and_result()
    manifest = ExportManifest.from_plan_and_result(plan, result)

    assert manifest.schema_version == "1.0"
    assert manifest.to_dict()["writer_name"] == "safetensors-dry-run"
    assert manifest.to_json() == manifest.to_json()


def test_write_manifest_creates_local_json(tmp_path: Path) -> None:
    plan, result = _plan_and_result()
    manifest = ExportManifest.from_plan_and_result(plan, result)
    manifest_path = tmp_path / "nested" / "export-manifest.json"

    bytes_written = write_export_manifest(manifest, manifest_path)

    assert bytes_written > 0
    assert manifest_path.read_text(encoding="utf-8") == manifest.to_json() + "\n"


def test_manifest_rejects_mismatched_result() -> None:
    plan, _ = _plan_and_result()
    mismatched = ExportWriteResult(
        output_path=Path("exports/other.safetensors"),
        export_format="safetensors",
        writer_name="safetensors-dry-run",
    )

    with pytest.raises(ValueError, match="output paths must match"):
        ExportManifest.from_plan_and_result(plan, mismatched)
