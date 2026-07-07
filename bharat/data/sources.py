from __future__ import annotations

from pathlib import Path

import yaml

from bharat.data.schema import DataSourceSpec


def load_source_spec(path: str | Path) -> DataSourceSpec:
    path = Path(path)
    file_path = str(path)

    if not path.exists():
        raise FileNotFoundError(f"Source spec file not found: {file_path}")

    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"{file_path}: source spec must be a YAML file (.yaml or .yml)")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{file_path}: malformed YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"{file_path}: YAML root must be a mapping, got {type(data).__name__}")

    return DataSourceSpec.from_dict(data, file_path)
