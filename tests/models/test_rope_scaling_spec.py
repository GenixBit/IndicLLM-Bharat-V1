from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.models.spec import load_model_spec

_SOURCE = Path(__file__).resolve().parents[2] / "configs/models/bharat-1b.yaml"


def test_production_rope_scaling_is_loaded() -> None:
    spec = load_model_spec(_SOURCE)
    assert spec.architecture.rope_scaling == {
        "type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 4096,
    }


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"type": "yarn", "factor": "eight"}, TypeError),
        ({"type": "yarn", "factor": 8.0, "extra": True}, ValueError),
        ({"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 0}, ValueError),
    ],
)
def test_invalid_rope_scaling_rejected(
    tmp_path: Path, override: dict[str, object], error: type[Exception]
) -> None:
    data = yaml.safe_load(_SOURCE.read_text(encoding="utf-8"))
    data["architecture"]["rope_scaling"] = override
    candidate = tmp_path / "model.yaml"
    candidate.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(error):
        load_model_spec(candidate)
