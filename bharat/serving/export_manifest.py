from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.serving.export import ExportPlan
from bharat.serving.export_writer import ExportWriteResult

EXPORT_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ExportManifest:
    checkpoint_path: Path
    output_path: Path
    export_format: str
    model_name: str
    dry_run: bool
    writer_name: str
    bytes_written: int
    schema_version: str = EXPORT_MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_plan_and_result(
        cls,
        plan: ExportPlan,
        result: ExportWriteResult,
    ) -> ExportManifest:
        if plan.output_path.resolve() != result.output_path.resolve():
            raise ValueError("plan and result output paths must match")
        if plan.export_format != result.export_format:
            raise ValueError("plan and result export formats must match")
        return cls(
            checkpoint_path=plan.checkpoint_path,
            output_path=plan.output_path,
            export_format=plan.export_format,
            model_name=plan.model_name,
            dry_run=result.dry_run,
            writer_name=result.writer_name,
            bytes_written=result.bytes_written,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes_written": self.bytes_written,
            "checkpoint_path": str(self.checkpoint_path),
            "dry_run": self.dry_run,
            "export_format": self.export_format,
            "model_name": self.model_name,
            "output_path": str(self.output_path),
            "schema_version": self.schema_version,
            "writer_name": self.writer_name,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def write_export_manifest(manifest: ExportManifest, manifest_path: Path) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_json() + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))
