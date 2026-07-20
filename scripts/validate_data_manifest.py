#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.manifest import DatasetManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest JSON file")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        errors.append(f"manifest file not found: {manifest_path}")
    elif manifest_path.suffix != ".json":
        errors.append(f"manifest must be a JSON file, got '{manifest_path.suffix}'")

    manifest: DatasetManifest | None = None
    if not errors:
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            manifest = DatasetManifest.from_dict(data)
        except json.JSONDecodeError as e:
            errors.append(f"invalid JSON: {e}")
        except ValueError as e:
            errors.append(str(e))

    if manifest is not None:
        manifest_errors = manifest.validate()
        errors.extend(manifest_errors)
        actual_digest = manifest.digest()
        if manifest.sha256 and actual_digest != manifest.sha256:
            warnings.append(
                f"manifest digest mismatch: declared '{manifest.sha256}', "
                f"computed '{actual_digest}'"
            )

    if args.json:
        result = {
            "status": "invalid" if errors else "valid",
            "manifest_path": str(manifest_path),
            "errors": errors,
            "warnings": warnings,
        }
        if manifest is not None:
            result["dataset_id"] = manifest.dataset_id
            result["records"] = manifest.records
            result["sha256"] = manifest.sha256
            if not manifest_errors:
                result["computed_digest"] = actual_digest
        print(json.dumps(result, indent=2))
    else:
        if not errors:
            print(f"Manifest valid: {manifest.dataset_id} ({manifest.records} records)")
            if warnings:
                for w in warnings:
                    print(f"  warning: {w}")
        else:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)
            for w in warnings:
                print(f"warning: {w}", file=sys.stderr)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
