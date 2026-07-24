from pathlib import Path

import pytest

from bharat.serving.export_inventory import build_checkpoint_inventory


def test_build_checkpoint_inventory_is_deterministic(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model":"bharat"}', encoding="utf-8")
    weights = checkpoint / "weights"
    weights.mkdir()
    (weights / "part-0001.bin").write_bytes(b"local-weights-fixture")

    first = build_checkpoint_inventory(checkpoint)
    second = build_checkpoint_inventory(checkpoint)

    assert first.to_json() == second.to_json()
    assert [item.relative_path for item in first.files] == [
        "config.json",
        "weights/part-0001.bin",
    ]
    assert first.total_bytes == sum(item.size_bytes for item in first.files)
    assert all(len(item.sha256) == 64 for item in first.files)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        build_checkpoint_inventory(tmp_path / "missing")


def test_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="must be a directory"):
        build_checkpoint_inventory(checkpoint)


def test_empty_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="contains no files"):
        build_checkpoint_inventory(checkpoint)
