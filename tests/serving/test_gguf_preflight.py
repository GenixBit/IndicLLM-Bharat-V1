import json
from pathlib import Path

import pytest

from bharat.serving.export_inventory import build_checkpoint_inventory
from bharat.serving.gguf_preflight import validate_gguf_preflight


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "bharat.gguf").write_bytes(b"local-gguf-fixture")
    metadata = tmp_path / "gguf-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": "bharat",
                "alignment": 32,
                "tensor_count": 2,
                "output_file": "bharat.gguf",
                "metadata": [
                    {"key": "general.name", "value_type": "string"},
                    {"key": "general.file_type", "value_type": "int"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, metadata


def test_validate_gguf_preflight_is_deterministic(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    inventory = build_checkpoint_inventory(checkpoint)

    first = validate_gguf_preflight(inventory, metadata)
    second = validate_gguf_preflight(inventory, metadata)

    assert first.to_json() == second.to_json()
    assert first.architecture == "bharat"
    assert first.alignment == 32
    assert first.tensor_count == 2
    assert [entry.key for entry in first.metadata] == [
        "general.file_type",
        "general.name",
    ]


def test_missing_output_file_is_rejected(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    inventory = build_checkpoint_inventory(checkpoint)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["output_file"] = "missing.gguf"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing output file"):
        validate_gguf_preflight(inventory, metadata)


def test_non_power_of_two_alignment_is_rejected(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    inventory = build_checkpoint_inventory(checkpoint)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["alignment"] = 24
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="positive power of two"):
        validate_gguf_preflight(inventory, metadata)


def test_duplicate_metadata_keys_are_rejected(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    inventory = build_checkpoint_inventory(checkpoint)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["metadata"] = [
        {"key": "general.name", "value_type": "string"},
        {"key": "general.name", "value_type": "string"},
    ]
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata keys must be unique"):
        validate_gguf_preflight(inventory, metadata)


def test_unsupported_metadata_value_type_is_rejected(tmp_path: Path) -> None:
    checkpoint, metadata = _fixture(tmp_path)
    inventory = build_checkpoint_inventory(checkpoint)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["metadata"] = [{"key": "general.name", "value_type": "object"}]
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported value_type"):
        validate_gguf_preflight(inventory, metadata)
