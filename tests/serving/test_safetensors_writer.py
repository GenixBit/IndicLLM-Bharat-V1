from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import torch

from bharat.serving.export_writer import ExportWriterRegistry
from bharat.serving.safetensors_writer import (
    SafetensorsWriteResult,
    _atomic_write,
    _build_metadata,
    _is_remote,
    _is_subpath,
    _load_state_dict,
    _resolve_checkpoint_pt_path,
    _validate_state_dict,
    write_safetensors_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_dict(
    tensors: dict[str, tuple[tuple[int, ...], torch.dtype, list[float]]],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(data, dtype=dtype).reshape(shape)
        for name, (shape, dtype, data) in tensors.items()
    }


def _write_pt(path: Path, obj: Any) -> None:
    torch.save(obj, path)


def _bharat_checkpoint(tmp_path: Path) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    _write_pt(
        cp / "model.pt",
        _make_state_dict(
            {
                "weight": ((2, 3), torch.float32, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                "bias": ((3,), torch.float32, [0.1, 0.2, 0.3]),
            }
        ),
    )
    return cp


def _training_checkpoint(tmp_path: Path) -> Path:
    pt = tmp_path / "training.pt"
    _write_pt(
        pt,
        {
            "model": _make_state_dict(
                {
                    "layer.weight": ((2, 2), torch.float32, [1.0, 0.0, 0.0, 1.0]),
                }
            ),
            "optimizer": {"step": 100},
        },
    )
    return pt


def _output_path(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    return out_dir / "bharat.safetensors"


# ---------------------------------------------------------------------------
# Unit: _is_remote
# ---------------------------------------------------------------------------


def test_is_remote_http() -> None:
    assert _is_remote(Path("http://example.com/model.pt"))


def test_is_remote_https() -> None:
    assert _is_remote(Path("https://example.com/model.pt"))


def test_is_remote_s3() -> None:
    assert _is_remote(Path("s3://bucket/model.pt"))


def test_is_remote_local_path() -> None:
    assert not _is_remote(Path("/home/user/model.pt"))
    assert not _is_remote(Path("./model.pt"))
    assert not _is_remote(Path("model.pt"))


# ---------------------------------------------------------------------------
# Unit: _is_subpath
# ---------------------------------------------------------------------------


def test_is_subpath_child() -> None:
    assert _is_subpath(Path("/a/b/c"), Path("/a"))


def test_is_subpath_not_child() -> None:
    assert not _is_subpath(Path("/a/b"), Path("/c"))


def test_is_subpath_same_path() -> None:
    assert _is_subpath(Path("/a"), Path("/a"))


# ---------------------------------------------------------------------------
# Unit: _resolve_checkpoint_pt_path
# ---------------------------------------------------------------------------


def test_resolve_directory_with_model_pt(tmp_path: Path) -> None:
    cp = tmp_path / "cp"
    cp.mkdir()
    (cp / "model.pt").write_text("dummy")
    result = _resolve_checkpoint_pt_path(cp)
    assert result == (cp / "model.pt").resolve()


def test_resolve_directory_missing_model_pt(tmp_path: Path) -> None:
    cp = tmp_path / "cp"
    cp.mkdir()
    with pytest.raises(FileNotFoundError, match="does not contain model.pt"):
        _resolve_checkpoint_pt_path(cp)


def test_resolve_pt_file(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_text("dummy")
    result = _resolve_checkpoint_pt_path(pt)
    assert result == pt.resolve()


def test_resolve_pth_file(tmp_path: Path) -> None:
    pth = tmp_path / "model.pth"
    pth.write_text("dummy")
    result = _resolve_checkpoint_pt_path(pth)
    assert result == pth.resolve()


def test_resolve_rejects_non_pt_file(tmp_path: Path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="must have a .pt or .pth extension"):
        _resolve_checkpoint_pt_path(txt)


def test_resolve_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_checkpoint_pt_path(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Unit: _load_state_dict
# ---------------------------------------------------------------------------


def test_load_direct_state_dict(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    sd = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    _write_pt(pt, sd)
    result = _load_state_dict(pt)
    assert set(result.keys()) == {"a", "b"}


def test_load_training_checkpoint(tmp_path: Path) -> None:
    pt = tmp_path / "ckpt.pt"
    _write_pt(
        pt,
        {
            "model": {"w": torch.tensor([[1.0]])},
            "optimizer": {"step": 10},
        },
    )
    result = _load_state_dict(pt)
    assert result == {"w": torch.tensor([[1.0]])}


def test_load_rejects_non_dict(tmp_path: Path) -> None:
    pt = tmp_path / "bad.pt"
    _write_pt(pt, [1, 2, 3])
    with pytest.raises(ValueError, match="unsupported checkpoint structure"):
        _load_state_dict(pt)


def test_load_rejects_missing_model_key(tmp_path: Path) -> None:
    pt = tmp_path / "ckpt.pt"
    _write_pt(pt, {"data": "not_tensors"})
    result = _load_state_dict(pt)
    assert isinstance(result, dict)
    assert result["data"] == "not_tensors"


# ---------------------------------------------------------------------------
# Unit: _validate_state_dict
# ---------------------------------------------------------------------------


def test_validate_empty_state_dict() -> None:
    with pytest.raises(ValueError, match="state dict is empty"):
        _validate_state_dict({})


def test_validate_empty_tensor_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid tensor name"):
        _validate_state_dict({"": torch.tensor(1.0)})


def test_validate_non_tensor_value() -> None:
    with pytest.raises(ValueError, match="expected torch.Tensor"):
        _validate_state_dict({"a": "not_a_tensor"})  # type: ignore[arg-type]


def test_validate_valid_state_dict() -> None:
    _validate_state_dict({"a": torch.tensor(1.0)})


# ---------------------------------------------------------------------------
# Unit: _build_metadata
# ---------------------------------------------------------------------------


def test_build_metadata_minimal() -> None:
    meta = _build_metadata(None, None)
    assert meta["format"] == "bharat-safetensors-v1"
    assert meta["writer_version"] == "1"


def test_build_metadata_with_model_name() -> None:
    meta = _build_metadata("bharat-350m", None)
    assert meta["model_name"] == "bharat-350m"


def test_build_metadata_with_caller_metadata() -> None:
    meta = _build_metadata(None, {"source": "test"})
    assert meta["source"] == "test"


def test_build_metadata_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="metadata key must be a non-empty string"):
        _build_metadata(None, {"": "value"})


def test_build_metadata_rejects_non_string_value() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        _build_metadata(None, {"key": 42})  # type: ignore[dict-item]


def test_build_metadata_rejects_reserved_key() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _build_metadata(None, {"format": "override"})


def test_build_metadata_is_deterministic() -> None:
    meta1 = _build_metadata("m1", {"z": "1", "a": "2"})
    meta2 = _build_metadata("m1", {"a": "2", "z": "1"})
    assert meta1 == meta2
    assert list(meta1.keys()) == sorted(meta1.keys())


# ---------------------------------------------------------------------------
# Unit: _atomic_write
# ---------------------------------------------------------------------------


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "model.safetensors"
    tensors = {"a": torch.tensor([1.0])}
    size = _atomic_write(tensors, {"format": "test"}, out)
    assert out.exists()
    assert size == out.stat().st_size
    assert size > 0


def test_atomic_write_rejects_concurrent_output(tmp_path: Path) -> None:
    out = tmp_path / "model.safetensors"
    out.write_text("existing")
    tensors = {"a": torch.tensor([1.0])}
    with pytest.raises(FileExistsError, match="created concurrently"):
        _atomic_write(tensors, {"format": "test"}, out)


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    out = tmp_path / "model.safetensors"

    class FailingTensor:
        def __init__(self) -> None:
            self.dtype = torch.float32

    tensors = {"a": FailingTensor()}  # type: ignore[arg-type]
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        _atomic_write(tensors, {"format": "test"}, out)
    assert not out.exists()
    temp_files = list(tmp_path.glob("*.safetensors"))
    assert len(temp_files) == 0


# ---------------------------------------------------------------------------
# Functional: write_safetensors_checkpoint
# ---------------------------------------------------------------------------

# --- 1. Writes a valid safetensors file ---


def test_writes_valid_safetensors_file(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(cp, out, model_name="bharat-local")
    assert out.exists()
    assert isinstance(result, SafetensorsWriteResult)


# --- 2. Output file is non-empty ---


def test_output_file_non_empty(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(cp, out)
    assert result.bytes_written > 0
    assert out.stat().st_size == result.bytes_written


# --- 3. Returned bytes_written equals output file size ---


def test_bytes_written_matches_file_size(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(cp, out)
    assert result.bytes_written == out.stat().st_size


# --- 4. Round-trip preserves tensor names ---


def test_roundtrip_preserves_tensor_names(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(cp, out)
    loaded = load_file(str(out))
    assert set(loaded.keys()) == {"weight", "bias"}


# --- 5. Round-trip preserves tensor shapes ---


def test_roundtrip_preserves_shapes(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(cp, out)
    loaded = load_file(str(out))
    assert loaded["weight"].shape == (2, 3)
    assert loaded["bias"].shape == (3,)


# --- 6. Round-trip preserves tensor dtypes ---


def test_roundtrip_preserves_dtypes(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    sd = _make_state_dict(
        {
            "a": ((2, 2), torch.float32, [1.0, 0.0, 0.0, 1.0]),
            "b": ((2,), torch.int64, [10, 20]),
        }
    )
    pt = tmp_path / "model.pt"
    _write_pt(pt, sd)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["a"].dtype == torch.float32
    assert loaded["b"].dtype == torch.int64


# --- 7. Round-trip preserves exact tensor values ---


def test_roundtrip_preserves_values(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    sd = _make_state_dict(
        {
            "w": ((2, 3), torch.float32, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
        }
    )
    pt = tmp_path / "model.pt"
    _write_pt(pt, sd)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert torch.equal(
        loaded["w"], torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32)
    )


# --- 8. Multiple tensors written deterministically ---


def test_multiple_tensors_deterministic(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    sd = _make_state_dict(
        {
            "z": ((1,), torch.float32, [1.0]),
            "a": ((1,), torch.float32, [2.0]),
            "m": ((1,), torch.float32, [3.0]),
        }
    )
    pt = tmp_path / "model.pt"
    _write_pt(pt, sd)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert set(loaded.keys()) == {"a", "m", "z"}


# --- 9. Tensor names are processed deterministically ---


def test_deterministic_tensor_order(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out1 = _output_path(tmp_path)
    out2 = tmp_path / "out2" / "bharat.safetensors"
    out2.parent.mkdir()
    r1 = write_safetensors_checkpoint(cp, out1)
    r2 = write_safetensors_checkpoint(cp, out2)
    assert r1.tensor_count == r2.tensor_count
    assert r1.metadata == r2.metadata


# --- 10. Metadata round-trips correctly ---


def test_metadata_roundtrip(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(
        cp, out, model_name="bharat-local", metadata={"source": "test"}
    )
    assert result.metadata["format"] == "bharat-safetensors-v1"
    assert result.metadata["model_name"] == "bharat-local"
    assert result.metadata["source"] == "test"
    assert result.metadata["writer_version"] == "1"


# --- 11. Empty tensor mapping is rejected ---


def test_rejects_empty_state_dict(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    _write_pt(pt, {})
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="state dict is empty"):
        write_safetensors_checkpoint(pt, out)


# --- 12. Non-tensor state-dict value is rejected ---


def test_rejects_non_tensor_value(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    _write_pt(pt, {"a": "string_value"})
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="expected torch.Tensor"):
        write_safetensors_checkpoint(pt, out)


# --- 13. Unsupported checkpoint structure is rejected ---


def test_rejects_unsupported_structure(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    _write_pt(pt, [1, 2, 3])
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="unsupported checkpoint structure"):
        write_safetensors_checkpoint(pt, out)


# --- 14. Missing checkpoint path is rejected ---


def test_rejects_missing_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent"
    out = _output_path(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_safetensors_checkpoint(missing, out)


# --- 15. Checkpoint path of the wrong type is rejected ---


def test_rejects_wrong_extension(tmp_path: Path) -> None:
    txt = tmp_path / "model.txt"
    txt.write_text("not a pt file")
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="must have a .pt or .pth extension"):
        write_safetensors_checkpoint(txt, out)


# --- 16. Missing output parent is rejected ---


def test_rejects_missing_output_parent(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = tmp_path / "nonexistent" / "model.safetensors"
    with pytest.raises(FileNotFoundError, match="output parent directory does not exist"):
        write_safetensors_checkpoint(cp, out)


# --- 17. Output parent that is a file is rejected ---


def test_rejects_output_parent_is_file(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    parent = tmp_path / "not_a_dir"
    parent.write_text("i am a file")
    out = parent / "model.safetensors"
    with pytest.raises(NotADirectoryError, match="output parent is not a directory"):
        write_safetensors_checkpoint(cp, out)


# --- 18. Existing output is rejected ---


def test_rejects_existing_output(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing file")
    with pytest.raises(FileExistsError, match="output path already exists"):
        write_safetensors_checkpoint(cp, out)


# --- 19. Output inside checkpoint directory is rejected ---


def test_rejects_output_inside_checkpoint(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = cp / "model.safetensors"
    with pytest.raises(ValueError, match="must not be inside the checkpoint directory"):
        write_safetensors_checkpoint(cp, out)


# --- 20. Unsupported tensor dtype produces a deterministic error ---


def test_unsupported_dtype_rejected(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    i = torch.tensor([[0, 0], [1, 1]])
    v = torch.tensor([1.0, 2.0], dtype=torch.float32)
    t = torch.sparse_coo_tensor(i, v, (2, 2))
    _write_pt(pt, {"a": t})
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="only strided"):
        write_safetensors_checkpoint(pt, out)


# --- 21. Non-contiguous tensors handled correctly ---


def test_non_contiguous_tensor(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pt = tmp_path / "model.pt"
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t = t[:, ::2]
    assert not t.is_contiguous()
    _write_pt(pt, {"a": t})
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert torch.equal(loaded["a"], t.contiguous())


# --- 22. Requires no GPU ---


def test_no_gpu_required(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(cp, out)
    assert result.tensor_count > 0


# --- 23. Does not mutate source tensors ---


def test_does_not_mutate_source(tmp_path: Path) -> None:
    sd = _make_state_dict(
        {
            "w": ((2, 2), torch.float32, [1.0, 2.0, 3.0, 4.0]),
        }
    )
    pt = tmp_path / "model.pt"
    _write_pt(pt, sd)
    original_values = {k: v.clone() for k, v in sd.items()}
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    for k, v in sd.items():
        assert torch.equal(v, original_values[k]), f"tensor {k} was mutated"


# --- 24. Failure removes temporary files ---


def test_failure_cleans_up_temp_files(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    temp_count_before = len(list(tmp_path.rglob("*.safetensors")))
    with suppress(FileNotFoundError, OSError):
        write_safetensors_checkpoint(cp, tmp_path / "nonexistent_dir" / "model.safetensors")
    temp_count_after = len(list(tmp_path.rglob("*.safetensors")))
    assert temp_count_after == temp_count_before


# --- 25. Failed write leaves no final output ---


def test_failed_write_no_output_file(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    _write_pt(pt, {"a": torch.tensor([1.0, 2.0])})
    out = tmp_path / "out" / "model.safetensors"
    with suppress(FileNotFoundError, OSError):
        write_safetensors_checkpoint(pt, out)
    assert not out.exists()


# --- 26. Remote-style paths are rejected ---


def test_rejects_remote_checkpoint(tmp_path: Path) -> None:
    out = _output_path(tmp_path)
    with pytest.raises(ValueError, match="checkpoint path must be local"):
        write_safetensors_checkpoint(Path("http://example.com/model.pt"), out)


def test_rejects_remote_output(tmp_path: Path) -> None:
    cp = _bharat_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="output path must be local"):
        write_safetensors_checkpoint(cp, Path("s3://bucket/model.safetensors"))


# --- 27. Existing dry-run writer tests remain unchanged and pass ---


def test_existing_dry_run_writer_still_works() -> None:
    from bharat.serving.export import ExportRequest, build_export_plan

    plan = build_export_plan(
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path("exports/bharat.safetensors"),
            export_format="safetensors",
            model_name="bharat-local",
        )
    )
    result = ExportWriterRegistry().write(plan)
    assert result.dry_run is True
    assert result.bytes_written == 0
    assert result.writer_name == "safetensors-dry-run"


# --- 28. Dry-run registry unchanged ---


def test_existing_dry_run_gguf_unchanged() -> None:
    from bharat.serving.export import ExportRequest, build_export_plan

    plan = build_export_plan(
        ExportRequest(
            checkpoint_path=Path("checkpoints/bharat"),
            output_path=Path("exports/bharat.gguf"),
            export_format="gguf",
            model_name="bharat-local",
        )
    )
    result = ExportWriterRegistry().write(plan)
    assert result.dry_run is True
    assert result.export_format == "gguf"
    assert result.writer_name == "gguf-dry-run"


# --- 29. Training checkpoint format is supported ---


def test_writes_from_training_checkpoint(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pt = _training_checkpoint(tmp_path)
    out = _output_path(tmp_path)
    result = write_safetensors_checkpoint(pt, out)
    assert out.exists()
    loaded = load_file(str(out))
    assert set(loaded.keys()) == {"layer.weight"}
    assert result.tensor_count == 1


# --- 30. .pth extension is supported ---


def test_supports_pth_extension(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    sd = _make_state_dict({"a": ((1,), torch.float32, [42.0])})
    pth = tmp_path / "model.pth"
    _write_pt(pth, sd)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pth, out)
    loaded = load_file(str(out))
    assert torch.equal(loaded["a"], torch.tensor([42.0], dtype=torch.float32))


# --- 31. bfloat16 support ---


def test_bfloat16_support(tmp_path: Path) -> None:
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 not supported in this torch version")
    from safetensors.torch import load_file

    pt = tmp_path / "model.pt"
    _write_pt(pt, {"a": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)})
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["a"].dtype == torch.bfloat16
    assert torch.equal(loaded["a"], torch.tensor([1.0, 2.0], dtype=torch.bfloat16))


# --- 32. bool support ---


def test_bool_tensor_support(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pt = tmp_path / "model.pt"
    _write_pt(pt, {"mask": torch.tensor([True, False, True])})
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["mask"].dtype == torch.bool
    assert torch.equal(loaded["mask"], torch.tensor([True, False, True]))


# --- 33. float16 support ---


def test_float16_support(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pt = tmp_path / "model.pt"
    _write_pt(pt, {"a": torch.tensor([1.0, 2.0], dtype=torch.float16)})
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["a"].dtype == torch.float16
    assert torch.equal(loaded["a"], torch.tensor([1.0, 2.0], dtype=torch.float16))


# --- 34. int64 support ---


def test_int64_support(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    pt = tmp_path / "model.pt"
    _write_pt(pt, {"a": torch.tensor([1, 2, 3], dtype=torch.int64)})
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["a"].dtype == torch.int64
    assert torch.equal(loaded["a"], torch.tensor([1, 2, 3], dtype=torch.int64))


# --- 35. Multiple dtypes in one checkpoint ---


def test_multiple_dtypes(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    sd = {
        "w": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "mask": torch.tensor([True, False], dtype=torch.bool),
        "ids": torch.tensor([10, 20], dtype=torch.int64),
    }
    pt = tmp_path / "model.pt"
    _write_pt(pt, sd)
    out = _output_path(tmp_path)
    write_safetensors_checkpoint(pt, out)
    loaded = load_file(str(out))
    assert loaded["w"].dtype == torch.float32
    assert loaded["mask"].dtype == torch.bool
    assert loaded["ids"].dtype == torch.int64
    assert loaded["w"].shape == (2, 2)
    assert loaded["mask"].shape == (2,)
    assert loaded["ids"].shape == (2,)
