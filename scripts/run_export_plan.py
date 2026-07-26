#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_inventory import build_checkpoint_inventory
from bharat.serving.export_manifest import ExportManifest, write_export_manifest
from bharat.serving.export_manifest_readiness import validate_export_manifest_readiness
from bharat.serving.export_path_readiness import validate_export_path_readiness
from bharat.serving.export_writer import ExportWriterRegistry
from bharat.serving.export_writer_readiness import validate_export_writer_readiness
from bharat.serving.gguf_preflight import validate_gguf_preflight
from bharat.serving.safetensors_preflight import validate_safetensors_preflight

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)
_NORMALIZED_URL_RE = re.compile(r"^(https?|ftp|s3|gs):/", re.IGNORECASE)


def _is_remote(path: str) -> bool:
    return bool(_URL_RE.match(path)) or bool(_NORMALIZED_URL_RE.match(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and run a local export plan for safetensors or GGUF",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["safetensors", "gguf"],
        help="Export format",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name for the export plan",
    )
    parser.add_argument(
        "--manifest-path",
        help="Optional local JSON manifest output path",
    )
    parser.add_argument(
        "--include-inventory",
        action="store_true",
        help="Include deterministic local checkpoint inventory metadata",
    )
    parser.add_argument(
        "--safetensors-metadata-path",
        help="Optional local safetensors metadata JSON to validate before export",
    )
    parser.add_argument(
        "--gguf-metadata-path",
        help="Optional local GGUF metadata JSON to validate before export",
    )
    parser.add_argument(
        "--validate-writer-readiness",
        action="store_true",
        help="Run writer readiness validation before the export dry-run",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute a real local export. Without this flag, the command remains a dry-run.",
    )
    parser.add_argument(
        "--gguf-tensor-type",
        default="f32",
        choices=["f32", "q8_0"],
        help="GGUF tensor type (default: f32)",
    )

    args = parser.parse_args()

    for label, value in (
        ("checkpoint", args.checkpoint_path),
        ("output", args.output_path),
        ("manifest", args.manifest_path),
        ("safetensors metadata", args.safetensors_metadata_path),
        ("GGUF metadata", args.gguf_metadata_path),
    ):
        if value is not None and _is_remote(value):
            print(f"error: Remote {label} path rejected: {value}", file=sys.stderr)
            sys.exit(1)

    if args.safetensors_metadata_path is not None and args.format != "safetensors":
        print(
            "error: --safetensors-metadata-path requires --format safetensors",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.gguf_metadata_path is not None and args.format != "gguf":
        print(
            "error: --gguf-metadata-path requires --format gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.gguf_tensor_type != "f32" and args.format != "gguf":
        print(
            "error: --gguf-tensor-type q8_0 requires --format gguf",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.execute and args.format == "gguf" and args.gguf_metadata_path is None:
        print(
            "error: --gguf-metadata-path is required for real GGUF execution",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.execute and args.gguf_tensor_type == "q8_0" and args.gguf_metadata_path is None:
        print(
            "error: --gguf-metadata-path is required for Q8_0 GGUF execution",
            file=sys.stderr,
        )
        sys.exit(1)

    dry_run = not args.execute

    try:
        request = ExportRequest(
            checkpoint_path=Path(args.checkpoint_path),
            output_path=Path(args.output_path),
            export_format=args.format,
            model_name=args.model_name,
            dry_run=dry_run,
            gguf_tensor_type=args.gguf_tensor_type,
        )
        plan = build_export_plan(request)

        metadata_paths = tuple(
            Path(path)
            for path in (
                args.safetensors_metadata_path,
                args.gguf_metadata_path,
            )
            if path is not None
        )
        has_path_targets = args.manifest_path is not None or len(metadata_paths) > 0
        export_path_readiness = None
        if has_path_targets:
            export_path_readiness = validate_export_path_readiness(
                plan,
                manifest_path=(None if args.manifest_path is None else Path(args.manifest_path)),
                metadata_paths=metadata_paths,
            )

        inventory = None
        needs_inventory = (
            args.include_inventory
            or args.safetensors_metadata_path is not None
            or args.gguf_metadata_path is not None
            or args.validate_writer_readiness
            or args.execute
        )
        if needs_inventory:
            inventory = build_checkpoint_inventory(plan.checkpoint_path)

        safetensors_preflight = None
        if inventory is not None and args.safetensors_metadata_path is not None:
            safetensors_preflight = validate_safetensors_preflight(
                inventory,
                Path(args.safetensors_metadata_path),
            )

        gguf_preflight = None
        if inventory is not None and args.gguf_metadata_path is not None:
            gguf_preflight = validate_gguf_preflight(
                inventory,
                Path(args.gguf_metadata_path),
                gguf_tensor_type=args.gguf_tensor_type,
            )

        writer_readiness = None
        if args.validate_writer_readiness or args.execute:
            writer_readiness = validate_export_writer_readiness(plan, inventory)

        manifest_readiness = None
        if args.manifest_path is not None:
            manifest_readiness = validate_export_manifest_readiness(
                plan,
                Path(args.manifest_path),
            )

        registry = ExportWriterRegistry(
            gguf_preflight=gguf_preflight,
            gguf_tensor_type=args.gguf_tensor_type,
        )
        result = registry.write(plan)

        if not plan.dry_run:
            if not result.output_path.exists():
                raise ValueError("export output not found after successful writer")
            actual_size = result.output_path.stat().st_size
            if result.bytes_written != actual_size:
                raise ValueError(
                    f"bytes_written {result.bytes_written} does not match "
                    f"actual output size {actual_size}"
                )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    output = {
        "checkpoint_path": str(plan.checkpoint_path),
        "output_path": str(plan.output_path),
        "export_format": plan.export_format,
        "model_name": plan.model_name,
        "dry_run": plan.dry_run,
        "writer_name": result.writer_name,
        "bytes_written": result.bytes_written,
    }

    if result.gguf_tensor_type is not None:
        output["gguf_tensor_type"] = result.gguf_tensor_type
    if result.f32_tensor_count is not None:
        output["f32_tensor_count"] = result.f32_tensor_count
    if result.q8_0_tensor_count is not None:
        output["q8_0_tensor_count"] = result.q8_0_tensor_count

    if export_path_readiness is not None:
        output["export_path_readiness"] = export_path_readiness.to_dict()

    if args.include_inventory and inventory is not None:
        output["checkpoint_inventory"] = inventory.to_dict()

    if safetensors_preflight is not None:
        output["safetensors_preflight"] = safetensors_preflight.to_dict()

    if gguf_preflight is not None:
        output["gguf_preflight"] = gguf_preflight.to_dict()

    if writer_readiness is not None:
        output["writer_readiness"] = writer_readiness.to_dict()

    if manifest_readiness is not None:
        output["manifest_readiness"] = manifest_readiness.to_dict()

    if args.manifest_path is not None:
        manifest = ExportManifest.from_plan_and_result(plan, result)
        manifest_path = Path(args.manifest_path)
        write_export_manifest(manifest, manifest_path)
        output["manifest_path"] = str(manifest_path)
        output["manifest_schema_version"] = manifest.schema_version

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
