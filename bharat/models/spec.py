from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bharat.models.config import BharatModelConfig

VALID_ARCH_KEYS: frozenset[str] = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "attention_dropout",
        "hidden_dropout",
        "initializer_range",
        "attention_bias",
        "mlp_bias",
        "tie_word_embeddings",
    }
)

VALID_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "model_name",
        "size_label",
        "target_parameter_count",
        "expected_parameter_count",
        "architecture",
    }
)


def _validate_int(value: Any, field: str, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path}: {field} must be an integer, got bool")
    if not isinstance(value, int):
        raise TypeError(f"{path}: {field} must be an integer, got {type(value).__name__}")
    return value


def _validate_float(value: Any, field: str, path: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path}: {field} must be a number, got bool")
    if not isinstance(value, int | float):
        raise TypeError(f"{path}: {field} must be a number, got {type(value).__name__}")
    return float(value)


def _validate_bool(value: Any, field: str, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path}: {field} must be a boolean, got {type(value).__name__}")
    return value


def _validate_str(value: Any, field: str, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: {field} must be a string, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class BharatModelSpec:
    schema_version: int
    model_name: str
    size_label: str
    target_parameter_count: int
    expected_parameter_count: int
    architecture: BharatModelConfig


def load_model_spec(path: str | Path) -> BharatModelSpec:
    path = Path(path)
    file_path = str(path)

    if not path.exists():
        raise FileNotFoundError(f"Model spec file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{file_path}: YAML root must be a mapping, got {type(data).__name__}")

    unknown_root = set(data) - VALID_ROOT_KEYS
    if unknown_root:
        raise ValueError(f"{file_path}: unknown root key(s): {', '.join(sorted(unknown_root))}")

    schema_version = _validate_int(data.get("schema_version"), "schema_version", file_path)
    if schema_version != 1:
        raise ValueError(f"{file_path}: unsupported schema_version {schema_version}, expected 1")

    model_name = _validate_str(data.get("model_name"), "model_name", file_path)
    if not model_name:
        raise ValueError(f"{file_path}: model_name must not be empty")

    size_label = _validate_str(data.get("size_label"), "size_label", file_path)
    if not size_label:
        raise ValueError(f"{file_path}: size_label must not be empty")

    target_count = _validate_int(
        data.get("target_parameter_count"), "target_parameter_count", file_path
    )
    if target_count <= 0:
        raise ValueError(
            f"{file_path}: target_parameter_count must be positive, got {target_count}"
        )

    expected_count = _validate_int(
        data.get("expected_parameter_count"), "expected_parameter_count", file_path
    )
    if expected_count <= 0:
        raise ValueError(
            f"{file_path}: expected_parameter_count must be positive, got {expected_count}"
        )

    arch_data = data.get("architecture")
    if not isinstance(arch_data, dict):
        raise ValueError(
            f"{file_path}: architecture must be a mapping, got {type(arch_data).__name__}"
        )

    unknown_arch = set(arch_data) - VALID_ARCH_KEYS
    if unknown_arch:
        raise ValueError(
            f"{file_path}: unknown architecture key(s): {', '.join(sorted(unknown_arch))}"
        )

    coerce = {
        "vocab_size": _validate_int(arch_data.get("vocab_size"), "vocab_size", file_path),
        "hidden_size": _validate_int(arch_data.get("hidden_size"), "hidden_size", file_path),
        "intermediate_size": _validate_int(
            arch_data.get("intermediate_size"), "intermediate_size", file_path
        ),
        "num_hidden_layers": _validate_int(
            arch_data.get("num_hidden_layers"), "num_hidden_layers", file_path
        ),
        "num_attention_heads": _validate_int(
            arch_data.get("num_attention_heads"), "num_attention_heads", file_path
        ),
        "num_key_value_heads": _validate_int(
            arch_data.get("num_key_value_heads"), "num_key_value_heads", file_path
        ),
        "max_position_embeddings": _validate_int(
            arch_data.get("max_position_embeddings"), "max_position_embeddings", file_path
        ),
        "rope_theta": _validate_float(arch_data.get("rope_theta"), "rope_theta", file_path),
        "rms_norm_eps": _validate_float(arch_data.get("rms_norm_eps"), "rms_norm_eps", file_path),
        "attention_dropout": _validate_float(
            arch_data.get("attention_dropout"), "attention_dropout", file_path
        ),
        "hidden_dropout": _validate_float(
            arch_data.get("hidden_dropout"), "hidden_dropout", file_path
        ),
        "initializer_range": _validate_float(
            arch_data.get("initializer_range"), "initializer_range", file_path
        ),
        "attention_bias": _validate_bool(
            arch_data.get("attention_bias"), "attention_bias", file_path
        ),
        "mlp_bias": _validate_bool(arch_data.get("mlp_bias"), "mlp_bias", file_path),
        "tie_word_embeddings": _validate_bool(
            arch_data.get("tie_word_embeddings"), "tie_word_embeddings", file_path
        ),
    }

    config = BharatModelConfig.from_dict(coerce)

    return BharatModelSpec(
        schema_version=schema_version,
        model_name=model_name,
        size_label=size_label,
        target_parameter_count=target_count,
        expected_parameter_count=expected_count,
        architecture=config,
    )


def load_model_config(path: str | Path) -> BharatModelConfig:
    return load_model_spec(path).architecture
