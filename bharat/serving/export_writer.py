from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from bharat.serving.export import ExportFormat, ExportPlan


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


class ExportWriterRegistry:
    def __init__(self, writers: tuple[ExportWriter, ...] | None = None) -> None:
        if writers is None:
            selected: tuple[ExportWriter, ...] = (  # type: ignore[assignment]
                DryRunExportWriter(name="safetensors-dry-run", export_format="safetensors"),
                DryRunExportWriter(name="gguf-dry-run", export_format="gguf"),
            )
        else:
            selected = writers
        self._writers: dict[ExportFormat, ExportWriter] = {}
        for writer in selected:
            if writer.export_format in self._writers:
                raise ValueError(f"duplicate writer for format {writer.export_format!r}")
            self._writers[writer.export_format] = writer

    def get(self, export_format: ExportFormat) -> ExportWriter:
        try:
            return self._writers[export_format]
        except KeyError as exc:
            raise ValueError(f"no writer registered for format {export_format!r}") from exc

    def write(self, plan: ExportPlan) -> ExportWriteResult:
        return self.get(plan.export_format).write(plan)
