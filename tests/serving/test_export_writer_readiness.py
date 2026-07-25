from __future__ import annotations

from pathlib import Path

import pytest

from bharat.serving.export import ExportPlan
from bharat.serving.export_inventory import CheckpointFile, CheckpointInventory
from bharat.serving.export_writer_readiness import validate_export_writer_readiness


def _plan(tmp_path: Path, checkpoint: Path, output: Path | None = None) -> ExportPlan:
    return ExportPlan(
        checkpoint_path=checkpoint,
        output_path=output or tmp_path / "out" / "bharat.safetensors",
        export_format="safetensors",
        model_name="bharat-local",
    )


def _inventory(checkpoint: Path, *, total_bytes: int = 4) -> CheckpointInventory:
    return CheckpointInventory(
        checkpoint_path=checkpoint,
        files=(
            CheckpointFile(
                relative_path="model.bin",
                size_bytes=4,
                sha256="0" * 64,
            ),
        ),
        total_bytes=total_bytes,
    )


def test_writer_readiness_is_deterministic_and_local(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_parent = tmp_path / "out"
    output_parent.mkdir()

    result = validate_export_writer_readiness(
        _plan(tmp_path, checkpoint),
        _inventory(checkpoint),
    )

    assert result.ready is True
    assert result.output_exists is False
    assert result.output_inside_checkpoint is False
    assert result.checkpoint_file_count == 1
    assert result.checkpoint_total_bytes == 4
    assert result.to_json() == result.to_json()


def test_writer_readiness_rejects_mismatched_inventory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "out").mkdir()

    with pytest.raises(ValueError, match="does not match export plan"):
        validate_export_writer_readiness(
            _plan(tmp_path, checkpoint),
            _inventory(other),
        )


def test_writer_readiness_rejects_inconsistent_inventory_total(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "out").mkdir()

    with pytest.raises(ValueError, match="total_bytes does not match"):
        validate_export_writer_readiness(
            _plan(tmp_path, checkpoint),
            _inventory(checkpoint, total_bytes=5),
        )


def test_writer_readiness_rejects_missing_output_parent(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="output parent directory does not exist"):
        validate_export_writer_readiness(
            _plan(tmp_path, checkpoint),
            _inventory(checkpoint),
        )


def test_writer_readiness_rejects_existing_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_parent = tmp_path / "out"
    output_parent.mkdir()
    output = output_parent / "bharat.safetensors"
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="output path already exists"):
        validate_export_writer_readiness(
            _plan(tmp_path, checkpoint, output),
            _inventory(checkpoint),
        )


def test_writer_readiness_rejects_output_inside_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = checkpoint / "bharat.safetensors"

    with pytest.raises(ValueError, match="must not be inside"):
        validate_export_writer_readiness(
            _plan(tmp_path, checkpoint, output),
            _inventory(checkpoint),
        )
