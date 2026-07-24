#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.serving.export import ExportRequest, build_export_plan
from bharat.serving.export_writer import ExportWriterRegistry

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)
_NORMALIZED_URL_RE = re.compile(r"^(https?|ftp|s3|gs):/", re.IGNORECASE)


def _is_remote(path: str) -> bool:
    return bool(_URL_RE.match(path)) or bool(_NORMALIZED_URL_RE.match(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and dry-run an export plan for safetensors or GGUF"
    )
    parser.add_argument("--checkpoint-path", required=True, help="Path to checkpoint directory")
    parser.add_argument("--output-path", required=True, help="Output file path")
    parser.add_argument(
        "--format",
        required=True,
        choices=["safetensors", "gguf"],
        help="Export format",
    )
    parser.add_argument("--model-name", required=True, help="Model name for the export plan")

    args = parser.parse_args()

    if _is_remote(args.checkpoint_path):
        print(f"error: Remote checkpoint path rejected: {args.checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if _is_remote(args.output_path):
        print(f"error: Remote output path rejected: {args.output_path}", file=sys.stderr)
        sys.exit(1)

    try:
        request = ExportRequest(
            checkpoint_path=Path(args.checkpoint_path),
            output_path=Path(args.output_path),
            export_format=args.format,
            model_name=args.model_name,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    plan = build_export_plan(request)

    try:
        registry = ExportWriterRegistry()
        result = registry.write(plan)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
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

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
