from pathlib import Path

import pytest

from bharat.serving.export import ExportRequest, build_export_plan


def test_build_safetensors_plan_is_deterministic() -> None:
    request = ExportRequest(
        checkpoint_path=Path("checkpoints/bharat"),
        output_path=Path("exports/bharat.safetensors"),
        export_format="safetensors",
        model_name="bharat-local",
    )

    plan = build_export_plan(request)

    assert plan.dry_run is True
    assert plan.to_dict()["export_format"] == "safetensors"
    assert plan.to_dict()["output_path"] == "exports/bharat.safetensors"


def test_build_gguf_plan() -> None:
    request = ExportRequest(
        checkpoint_path=Path("checkpoints/bharat"),
        output_path=Path("exports/bharat.gguf"),
        export_format="gguf",
        model_name="bharat-local",
    )

    assert build_export_plan(request).export_format == "gguf"


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/model.gguf",
        "s3://bucket/model.gguf",
        "gs://bucket/model.gguf",
    ],
)
def test_remote_output_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="local filesystem path"):
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path(path),
            export_format="gguf",
            model_name="bharat-local",
        )


def test_wrong_suffix_is_rejected() -> None:
    with pytest.raises(ValueError, match="must end with"):
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path("exports/bharat.bin"),
            export_format="safetensors",
            model_name="bharat-local",
        )


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_name"):
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path("exports/bharat.gguf"),
            export_format="gguf",
            model_name=" ",
        )
