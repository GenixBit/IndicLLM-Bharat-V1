import json
from pathlib import Path

import pytest

from bharat.serving.export_inventory import build_checkpoint_inventory
from bharat.serving.safetensors_preflight import validate_safetensors_preflight


def _checkpoint(tmp_path: Path) -> tuple[Path, object]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"local-shard-fixture")
    return checkpoint, build_checkpoint_inventory(checkpoint)


def _write_metadata(path: Path, tensors: list[dict[str, object]], total: int) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_tensor_bytes": total,
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )


def test_preflight_is_deterministic_and_sorts_tensors(tmp_path: Path) -> None:
    _, inventory = _checkpoint(tmp_path)
    metadata = tmp_path / "metadata.json"
    _write_metadata(
        metadata,
        [
            {
                "name": "transformer.z.weight",
                "shape": [2, 4],
                "dtype": "F32",
                "shard": "model-00001-of-00001.safetensors",
                "size_bytes": 32,
            },
            {
                "name": "transformer.a.weight",
                "shape": [4],
                "dtype": "BF16",
                "shard": "model-00001-of-00001.safetensors",
                "size_bytes": 8,
            },
        ],
        total=40,
    )

    first = validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]
    second = validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]

    assert first.to_json() == second.to_json()
    assert [tensor.name for tensor in first.tensors] == [
        "transformer.a.weight",
        "transformer.z.weight",
    ]
    assert first.tensor_count == 2
    assert first.total_tensor_bytes == 40


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    _, inventory = _checkpoint(tmp_path)
    metadata = tmp_path / "metadata.json"
    _write_metadata(
        metadata,
        [
            {
                "name": "weight",
                "shape": [1],
                "dtype": "F32",
                "shard": "missing.safetensors",
                "size_bytes": 4,
            }
        ],
        total=4,
    )

    with pytest.raises(ValueError, match="missing shards"):
        validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]


def test_duplicate_tensor_names_are_rejected(tmp_path: Path) -> None:
    _, inventory = _checkpoint(tmp_path)
    metadata = tmp_path / "metadata.json"
    tensor = {
        "name": "weight",
        "shape": [1],
        "dtype": "F32",
        "shard": "model-00001-of-00001.safetensors",
        "size_bytes": 4,
    }
    _write_metadata(metadata, [tensor, tensor], total=8)

    with pytest.raises(ValueError, match="unique"):
        validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]


def test_declared_total_must_match_tensor_sum(tmp_path: Path) -> None:
    _, inventory = _checkpoint(tmp_path)
    metadata = tmp_path / "metadata.json"
    _write_metadata(
        metadata,
        [
            {
                "name": "weight",
                "shape": [1],
                "dtype": "F32",
                "shard": "model-00001-of-00001.safetensors",
                "size_bytes": 4,
            }
        ],
        total=8,
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]


def test_invalid_shape_and_dtype_are_rejected(tmp_path: Path) -> None:
    _, inventory = _checkpoint(tmp_path)
    metadata = tmp_path / "metadata.json"
    _write_metadata(
        metadata,
        [
            {
                "name": "weight",
                "shape": [0],
                "dtype": "UNKNOWN",
                "shard": "model-00001-of-00001.safetensors",
                "size_bytes": 0,
            }
        ],
        total=0,
    )

    with pytest.raises(ValueError, match="shape dimensions"):
        validate_safetensors_preflight(inventory, metadata)  # type: ignore[arg-type]
