from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bharat.serving.export import ExportFormat, ExportPlan
from bharat.serving.safetensors_writer import write_safetensors_checkpoint


@dataclass(frozen=True)
class ExportWriteResult:
    output_path: Path
    export_format: ExportFormat
    writer_name: str
    dry_run: bool = True
    bytes_written: int = 0

    def __post_init__(self) -> None:
        if self.bytes_written < 0:
            raise ValueError("bytes_written must be >= 0")
        if self.dry_run and self.bytes_written != 0:
            raise ValueError("dry-run results must report zero bytes written")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "export_format": self.export_format,
            "writer_name": self.writer_name,
            "dry_run": self.dry_run,
            "bytes_written": self.bytes_written,
        }

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
        return ExportWriteResult(
            output_path=plan.output_path,
            export_format=plan.export_format,
            writer_name=self.name,
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


class ExportWriterRegistry:
    def __init__(self, writers: tuple[ExportWriter, ...] | None = None) -> None:
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
