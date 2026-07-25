from __future__ import annotations

from pathlib import Path

import pytest

from bharat.serving.export import ExportPlan
from bharat.serving.export_path_readiness import validate_export_path_readiness


def _plan(tmp_path: Path) -> ExportPlan:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    return ExportPlan(
        checkpoint_path=checkpoint,
        output_path=output_parent / "bharat.safetensors",
        export_format="safetensors",
        model_name="bharat-local",
    )


def test_path_readiness_is_deterministic_and_sorted(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    manifest = tmp_path / "out" / "manifest.json"
    metadata_b = tmp_path / "metadata-b.json"
    metadata_a = tmp_path / "metadata-a.json"
    metadata_b.write_text("{}", encoding="utf-8")
    metadata_a.write_text("{}", encoding="utf-8")

    result = validate_export_path_readiness(
        plan,
        manifest_path=manifest,
        metadata_paths=(metadata_b, metadata_a),
    )

    assert result.ready is True
    assert result.metadata_paths == (metadata_a.resolve(), metadata_b.resolve())
    assert result.to_json() == result.to_json()


def test_path_readiness_rejects_duplicate_metadata_paths(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        validate_export_path_readiness(plan, metadata_paths=(metadata, metadata))


def test_path_readiness_rejects_missing_metadata_path(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(ValueError, match="does not exist"):
        validate_export_path_readiness(
            plan,
            metadata_paths=(tmp_path / "missing.json",),
        )


def test_path_readiness_rejects_metadata_directory(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    metadata_directory = tmp_path / "metadata"
    metadata_directory.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        validate_export_path_readiness(plan, metadata_paths=(metadata_directory,))


def test_path_readiness_rejects_metadata_output_collision(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan.output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not equal export output"):
        validate_export_path_readiness(plan, metadata_paths=(plan.output_path,))


def test_path_readiness_rejects_metadata_manifest_collision(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not equal export manifest"):
        validate_export_path_readiness(
            plan,
            manifest_path=manifest,
            metadata_paths=(manifest,),
        )
