from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as _safetensors_save_file

_WRITER_VERSION = "1"
_FORMAT_ID = "bharat-safetensors-v1"

_RESERVED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "format",
        "writer_version",
    }
)

_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")


@dataclass(frozen=True)
class SafetensorsWriteResult:
    output_path: Path
    tensor_count: int
    bytes_written: int
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "tensor_count": self.tensor_count,
            "bytes_written": self.bytes_written,
            "metadata": dict(sorted(self.metadata.items())),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _is_remote(path: Path) -> bool:
    raw = str(path).lower()
    for prefix in _REMOTE_PREFIXES:
        if raw.startswith(prefix) or raw.startswith(prefix.replace("://", ":/")):
            return True
    return False


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_checkpoint_pt_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        pt_path = checkpoint_path / "model.pt"
        if not pt_path.is_file():
            raise FileNotFoundError(
                f"checkpoint directory {checkpoint_path} does not contain model.pt"
            )
        return pt_path
    if checkpoint_path.is_file():
        if checkpoint_path.suffix.lower() not in (".pt", ".pth"):
            raise ValueError(
                f"checkpoint file must have a .pt or .pth extension: {checkpoint_path}"
            )
        return checkpoint_path
    raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")


def _load_state_dict(pt_path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"failed to load checkpoint {pt_path}: {exc}. "
            "Only PyTorch state dicts loadable with weights_only=True are supported."
        ) from exc

    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj

    raise ValueError(f"unsupported checkpoint structure in {pt_path}. " "Expected a dict.")


def _validate_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    if not state_dict:
        raise ValueError("state dict is empty")
    for name in state_dict:
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid tensor name: {name!r}")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"value for key {name!r} is {type(tensor).__name__}, expected torch.Tensor"
            )
        if tensor.is_sparse or tensor.layout != torch.strided:
            raise ValueError(
                f"tensor {name!r} has layout {tensor.layout}; "
                f"only strided (dense) tensors are supported"
            )


def _build_metadata(
    model_name: str | None,
    caller_metadata: Mapping[str, str] | None,
) -> dict[str, str]:
    meta: dict[str, str] = {
        "format": _FORMAT_ID,
        "writer_version": _WRITER_VERSION,
    }
    if model_name:
        meta["model_name"] = model_name

    if caller_metadata:
        for key, value in caller_metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"metadata key must be a non-empty string, got {key!r}")
            if not isinstance(value, str):
                raise ValueError(
                    f"metadata value for {key!r} must be a string, " f"got {type(value).__name__}"
                )
            if key in _RESERVED_METADATA_KEYS:
                raise ValueError(f"metadata key {key!r} is reserved and cannot be overridden")
            meta[key] = value

    return dict(sorted(meta.items()))


def _atomic_write(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    output_path: Path,
) -> int:
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_path.parent),
            suffix=".safetensors",
        )
        os.close(fd)

        try:
            _safetensors_save_file(tensors, tmp_path, metadata=metadata)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise ValueError(f"unsupported tensor dtype or structure: {exc}") from exc

        file_size = os.path.getsize(tmp_path)
        if file_size == 0:
            raise RuntimeError("temporary safetensors file is empty after write")

        if output_path.exists():
            raise FileExistsError(f"output path was created concurrently: {output_path}")

        try:
            os.link(tmp_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"output path was created concurrently: {output_path}"
            ) from exc
        os.unlink(tmp_path)
        tmp_path = None
        return file_size
    except BaseException:
        if tmp_path is not None and os.path.exists(tmp_path):
            with suppress(OSError):
                os.unlink(tmp_path)
        raise


def write_safetensors_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    model_name: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> SafetensorsWriteResult:
    if _is_remote(checkpoint_path):
        raise ValueError(f"checkpoint path must be local: {checkpoint_path}")
    if _is_remote(output_path):
        raise ValueError(f"output path must be local: {output_path}")

    resolved_checkpoint = checkpoint_path.resolve()
    resolved_output = output_path.resolve()

    pt_path = _resolve_checkpoint_pt_path(resolved_checkpoint)

    if resolved_output.exists():
        raise FileExistsError(f"output path already exists: {resolved_output}")
    if not resolved_output.parent.exists():
        raise FileNotFoundError(f"output parent directory does not exist: {resolved_output.parent}")
    if not resolved_output.parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {resolved_output.parent}")

    if resolved_checkpoint.is_dir() and _is_subpath(resolved_output, resolved_checkpoint):
        raise ValueError(
            f"output path must not be inside the checkpoint directory: " f"{resolved_output}"
        )

    state_dict = _load_state_dict(pt_path)
    _validate_state_dict(state_dict)

    writer_metadata = _build_metadata(model_name, metadata)

    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        tensors[name] = t

    tensors = dict(sorted(tensors.items()))

    bytes_written = _atomic_write(tensors, writer_metadata, resolved_output)

    return SafetensorsWriteResult(
        output_path=resolved_output,
        tensor_count=len(tensors),
        bytes_written=bytes_written,
        metadata=writer_metadata,
    )
