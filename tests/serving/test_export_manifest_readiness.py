from __future__ import annotations

from pathlib import Path

import pytest

from bharat.serving.export import ExportPlan
from bharat.serving.export_manifest_readiness import validate_export_manifest_readiness


def _plan(tmp_path: Path, checkpoint: Path) -> ExportPlan:
    return ExportPlan(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "out" / "bharat.safetensors",
        export_format="safetensors",
        model_name="bharat-local",
    )


def test_manifest_readiness_is_deterministic_and_local(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    manifest_parent = tmp_path / "manifests"
    manifest_parent.mkdir()

    result = validate_export_manifest_readiness(
        _plan(tmp_path, checkpoint),
        manifest_parent / "export.json",
    )

    assert result.ready is True
    assert result.manifest_exists is False
    assert result.manifest_conflicts_with_output is False
    assert result.manifest_inside_checkpoint is False
    assert result.to_json() == result.to_json()
    assert not result.manifest_path.exists()


def test_manifest_readiness_rejects_missing_parent(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "out").mkdir()

    with pytest.raises(ValueError, match="manifest parent directory does not exist"):
        validate_export_manifest_readiness(
            _plan(tmp_path, checkpoint),
            tmp_path / "missing" / "export.json",
        )


def test_manifest_readiness_rejects_parent_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "out").mkdir()
    parent = tmp_path / "manifest-parent"
    parent.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest parent path must be a directory"):
        validate_export_manifest_readiness(
            _plan(tmp_path, checkpoint),
            parent / "export.json",
        )


def test_manifest_readiness_rejects_existing_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "out").mkdir()
    manifest = tmp_path / "export.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest path already exists"):
        validate_export_manifest_readiness(_plan(tmp_path, checkpoint), manifest)


def test_manifest_readiness_rejects_output_collision(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    plan = _plan(tmp_path, checkpoint)

    with pytest.raises(ValueError, match="must not equal export output path"):
        validate_export_manifest_readiness(plan, plan.output_path)


def test_manifest_readiness_rejects_path_inside_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "out").mkdir()

    with pytest.raises(ValueError, match="must not be inside"):
        validate_export_manifest_readiness(
            _plan(tmp_path, checkpoint),
            checkpoint / "export.json",
        )
