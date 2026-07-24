from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

ExportFormat = Literal["safetensors", "gguf"]
_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")
_SUFFIXES: dict[ExportFormat, str] = {
    "safetensors": ".safetensors",
    "gguf": ".gguf",
}


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    export_format: ExportFormat
    model_name: str
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "export_format": self.export_format,
            "model_name": self.model_name,
            "success": self.success,
            "message": self.message,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class ExportWriter(Protocol):
    def __init__(self, export_format: str) -> None: ...

    def write(self, plan: ExportPlan) -> ExportResult: ...


class DryRunExportWriter:
    def __init__(self, export_format: ExportFormat) -> None:
        self.export_format = export_format

    def write(self, plan: ExportPlan) -> ExportResult:
        valid_suffix = _SUFFIXES[plan.export_format]
        if plan.output_path.suffix.lower() != valid_suffix:
            return ExportResult(
                output_path=plan.output_path,
                export_format=plan.export_format,
                model_name=plan.model_name,
                success=False,
                message=f"output_path must end with {valid_suffix!r}",
            )
        return ExportResult(
            output_path=plan.output_path,
            export_format=plan.export_format,
            model_name=plan.model_name,
            success=True,
            message=f"Dry-run export of {plan.model_name} to {plan.output_path} [{plan.export_format}]",
        )


_WRITER_REGISTRY: dict[ExportFormat, type[ExportWriter]] = {}


def register_writer(export_format: ExportFormat, writer_cls: type[ExportWriter]) -> None:
    _WRITER_REGISTRY[export_format] = writer_cls


def get_writer(export_format: ExportFormat) -> ExportWriter:
    try:
        cls = _WRITER_REGISTRY[export_format]
    except KeyError:
        raise ValueError(f"No writer registered for export format: {export_format!r}") from None
    return cls(export_format)


def run_export(plan: ExportPlan) -> ExportResult:
    writer = get_writer(plan.export_format)
    return writer.write(plan)


register_writer("safetensors", DryRunExportWriter)
register_writer("gguf", DryRunExportWriter)


def _validate_local_path(path: Path, label: str) -> None:
    raw = str(path)
    lowered = raw.lower()
    if lowered.startswith(_REMOTE_PREFIXES):
        raise ValueError(f"{label} must be a local filesystem path")
    for prefix in _REMOTE_PREFIXES:
        normalized_prefix = prefix.replace("://", ":/")
        if lowered.startswith(normalized_prefix):
            raise ValueError(f"{label} must be a local filesystem path")


@dataclass(frozen=True)
class ExportRequest:
    checkpoint_path: Path
    output_path: Path
    export_format: ExportFormat
    model_name: str

    def __post_init__(self) -> None:
        _validate_local_path(self.checkpoint_path, "checkpoint_path")
        _validate_local_path(self.output_path, "output_path")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        expected_suffix = _SUFFIXES[self.export_format]
        if self.output_path.suffix.lower() != expected_suffix:
            raise ValueError(
                f"output_path for {self.export_format!r} must end with {expected_suffix!r}"
            )


@dataclass(frozen=True)
class ExportPlan:
    checkpoint_path: Path
    output_path: Path
    export_format: ExportFormat
    model_name: str
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "output_path": str(self.output_path),
            "export_format": self.export_format,
            "model_name": self.model_name,
            "dry_run": self.dry_run,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_export_plan(request: ExportRequest) -> ExportPlan:
    """Build a deterministic local export plan without reading or writing weights."""
    return ExportPlan(
        checkpoint_path=request.checkpoint_path,
        output_path=request.output_path,
        export_format=request.export_format,
        model_name=request.model_name,
    )
