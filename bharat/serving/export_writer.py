from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from bharat.serving.export import GGUF_TENSOR_TYPE_VALUES, ExportFormat, ExportPlan
from bharat.serving.gguf_preflight import GGUFPreflightResult
from bharat.serving.gguf_tensor_writer import write_gguf_f32_tensors, write_gguf_q8_0_tensors
from bharat.serving.safetensors_writer import write_safetensors_checkpoint


@dataclass(frozen=True)
class ExportWriteResult:
    output_path: Path
    export_format: ExportFormat
    writer_name: str
    dry_run: bool = True
    bytes_written: int = 0
    gguf_tensor_type: str | None = None
    f32_tensor_count: int | None = None
    q8_0_tensor_count: int | None = None

    def __post_init__(self) -> None:
        if self.bytes_written < 0:
            raise ValueError("bytes_written must be >= 0")
        if self.dry_run and self.bytes_written != 0:
            raise ValueError("dry-run results must report zero bytes written")
        if (
            self.gguf_tensor_type is not None
            and self.gguf_tensor_type not in GGUF_TENSOR_TYPE_VALUES
        ):
            raise ValueError(
                f"gguf_tensor_type must be one of {sorted(GGUF_TENSOR_TYPE_VALUES)}, "
                f"got {self.gguf_tensor_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "output_path": str(self.output_path),
            "export_format": self.export_format,
            "writer_name": self.writer_name,
            "dry_run": self.dry_run,
            "bytes_written": self.bytes_written,
        }
        if self.gguf_tensor_type is not None:
            d["gguf_tensor_type"] = self.gguf_tensor_type
        if self.f32_tensor_count is not None:
            d["f32_tensor_count"] = self.f32_tensor_count
        if self.q8_0_tensor_count is not None:
            d["q8_0_tensor_count"] = self.q8_0_tensor_count
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class ExportWriter(Protocol):
    name: str
    export_format: ExportFormat

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        """Execute a local export plan."""


@dataclass(frozen=True)
class DryRunExportWriter:
    name: str
    export_format: ExportFormat

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        if plan.export_format != self.export_format:
            raise ValueError(f"writer {self.name!r} does not support format {plan.export_format!r}")
        if not plan.dry_run:
            raise ValueError("dry-run writer requires a dry-run export plan")
        kwargs: dict[str, Any] = {}
        if plan.export_format == "gguf":
            kwargs["gguf_tensor_type"] = plan.gguf_tensor_type
        return ExportWriteResult(
            output_path=plan.output_path,
            export_format=plan.export_format,
            writer_name=self.name,
            **kwargs,
        )


@dataclass(frozen=True)
class LocalSafetensorsExportWriter:
    name: str = "safetensors-local"
    export_format: ExportFormat = "safetensors"

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        if plan.export_format != self.export_format:
            raise ValueError(f"writer {self.name!r} does not support format {plan.export_format!r}")
        if plan.dry_run:
            raise ValueError("real safetensors writer requires a non-dry-run export plan")
        result = write_safetensors_checkpoint(
            checkpoint_path=plan.checkpoint_path,
            output_path=plan.output_path,
            model_name=plan.model_name,
        )
        return ExportWriteResult(
            output_path=result.output_path,
            export_format=self.export_format,
            writer_name=self.name,
            dry_run=False,
            bytes_written=result.bytes_written,
        )


def _resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    resolved = checkpoint_path.resolve()
    if resolved.is_dir():
        model_path = resolved / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"checkpoint directory {resolved} does not contain model.pt")
        return model_path
    if resolved.is_file() and resolved.suffix.lower() in (".pt", ".pth"):
        return resolved
    raise FileNotFoundError(f"checkpoint path is not a local .pt/.pth file: {resolved}")


def _load_f32_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    model_path = _resolve_checkpoint_file(checkpoint_path)
    try:
        loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"failed to load checkpoint {model_path} with weights_only=True: {exc}"
        ) from exc

    state_dict: object = (
        loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
    )
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain a state-dict mapping")
    return {str(name): tensor for name, tensor in state_dict.items()}


def _validate_gguf_tensor_type(
    *,
    writer_name: str,
    expected: str,
    plan: ExportPlan,
    preflight: GGUFPreflightResult,
) -> None:
    if plan.gguf_tensor_type != expected:
        raise ValueError(
            f"writer {writer_name!r} requires gguf_tensor_type {expected!r}, "
            f"got {plan.gguf_tensor_type!r}"
        )
    if preflight.gguf_tensor_type != expected:
        raise ValueError(
            f"writer {writer_name!r} requires preflight gguf_tensor_type {expected!r}, "
            f"got {preflight.gguf_tensor_type!r}"
        )


@dataclass(frozen=True)
class LocalGGUFF32ExportWriter:
    preflight: GGUFPreflightResult
    name: str = "gguf-f32-local"
    export_format: ExportFormat = "gguf"

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        if plan.export_format != self.export_format:
            raise ValueError(f"writer {self.name!r} does not support format {plan.export_format!r}")
        if plan.dry_run:
            raise ValueError("real GGUF writer requires a non-dry-run export plan")
        _validate_gguf_tensor_type(
            writer_name=self.name,
            expected="f32",
            plan=plan,
            preflight=self.preflight,
        )
        tensors = _load_f32_state_dict(plan.checkpoint_path)
        result = write_gguf_f32_tensors(self.preflight, tensors, plan.output_path.resolve())
        return ExportWriteResult(
            output_path=result.output_path,
            export_format=self.export_format,
            writer_name=self.name,
            dry_run=False,
            bytes_written=result.bytes_written,
            gguf_tensor_type="f32",
            f32_tensor_count=result.tensor_count,
            q8_0_tensor_count=0,
        )


@dataclass(frozen=True)
class LocalGGUFQ8_0ExportWriter:  # noqa: N801
    preflight: GGUFPreflightResult
    name: str = "gguf-q8_0-local"
    export_format: ExportFormat = "gguf"

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        if plan.export_format != self.export_format:
            raise ValueError(f"writer {self.name!r} does not support format {plan.export_format!r}")
        if plan.dry_run:
            raise ValueError("real GGUF writer requires a non-dry-run export plan")
        _validate_gguf_tensor_type(
            writer_name=self.name,
            expected="q8_0",
            plan=plan,
            preflight=self.preflight,
        )
        tensors = _load_f32_state_dict(plan.checkpoint_path)
        result = write_gguf_q8_0_tensors(self.preflight, tensors, plan.output_path.resolve())
        return ExportWriteResult(
            output_path=result.output_path,
            export_format=self.export_format,
            writer_name=self.name,
            dry_run=False,
            bytes_written=result.bytes_written,
            gguf_tensor_type="q8_0",
            f32_tensor_count=0,
            q8_0_tensor_count=result.tensor_count,
        )


class ExportWriterRegistry:
    def __init__(
        self,
        writers: tuple[ExportWriter, ...] | None = None,
        *,
        gguf_preflight: GGUFPreflightResult | None = None,
        gguf_tensor_type: str = "f32",
    ) -> None:
        if gguf_tensor_type not in GGUF_TENSOR_TYPE_VALUES:
            raise ValueError(
                f"unsupported GGUF tensor type: {gguf_tensor_type!r}; "
                f"expected one of {sorted(GGUF_TENSOR_TYPE_VALUES)}"
            )
        if (
            gguf_preflight is not None
            and gguf_preflight.gguf_tensor_type != gguf_tensor_type
        ):
            raise ValueError(
                "GGUF preflight tensor type does not match registry selection: "
                f"{gguf_preflight.gguf_tensor_type!r} != {gguf_tensor_type!r}"
            )

        self._writers: dict[tuple[ExportFormat, bool], ExportWriter] = {}
        if writers is None:
            self._writers[("safetensors", True)] = DryRunExportWriter(  # type: ignore[assignment]
                name="safetensors-dry-run",
                export_format="safetensors",
            )
            self._writers[("safetensors", False)] = LocalSafetensorsExportWriter()  # type: ignore[assignment]
            self._writers[("gguf", True)] = DryRunExportWriter(  # type: ignore[assignment]
                name="gguf-dry-run",
                export_format="gguf",
            )
            if gguf_preflight is not None:
                if gguf_tensor_type == "f32":
                    self._writers[("gguf", False)] = LocalGGUFF32ExportWriter(  # type: ignore[assignment]
                        preflight=gguf_preflight,
                    )
                else:
                    self._writers[("gguf", False)] = LocalGGUFQ8_0ExportWriter(  # type: ignore[assignment]
                        preflight=gguf_preflight,
                    )
        else:
            for writer in writers:
                key = (writer.export_format, True)
                if key in self._writers:
                    raise ValueError(f"duplicate writer for format {writer.export_format!r}")
                self._writers[key] = writer

    def get(self, export_format: ExportFormat, dry_run: bool = True) -> ExportWriter:
        key = (export_format, dry_run)
        try:
            return self._writers[key]
        except KeyError as exc:
            if not dry_run:
                raise ValueError(
                    f"no execute writer registered for format {export_format!r}"
                ) from exc
            raise ValueError(f"no writer registered for format {export_format!r}") from exc

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        return self.get(plan.export_format, dry_run=plan.dry_run).write(plan)
