#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Dataset Manifest Validator CLI.

Validates data manifest integrity, cryptographic digests, schema compliance,
and shard coverage.

Usage:
  python scripts/validate_data_manifest.py --manifest data/governed/sangraha_hi/manifest.json
  python scripts/validate_data_manifest.py --manifest data/governed/sangraha_hi/manifest.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.manifest import DatasetManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest file")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON result")

    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)

    errors: list[str] = []
    warnings: list[str] = []
    manifest: DatasetManifest | None = None

    if not manifest_path.exists():
        errors.append(f"manifest file not found: {manifest_path}")
    else:
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            manifest = DatasetManifest.from_dict(data)
        except json.JSONDecodeError as e:
            errors.append(f"invalid JSON: {e}")
        except Exception as e:
            errors.append(f"manifest parse error: {e}")

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
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print(f"  🇮🇳 Dataset Manifest Validation: {manifest_path.name}")
        print("=" * 60)
        if errors:
            print(f"  Status: INVALID ({len(errors)} errors)")
            for e in errors:
                print(f"    - {e}")
            print(
                "error: validation failed:\n" + "\n".join(f"  - {e}" for e in errors),
                file=sys.stderr,
            )
        else:
            print("  Status: VALID")
            if manifest:
                print(f"  Manifest valid: {manifest.dataset_id}")
                print(f"  Dataset: {manifest.dataset_id}")
                print(f"  Records: {manifest.records:,}")
                print(f"  Shards : {len(manifest.shards)}")
                print(f"  SHA-256: {manifest.sha256}")
        if warnings:
            print(f"  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    - {w}")
        print("=" * 60)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
