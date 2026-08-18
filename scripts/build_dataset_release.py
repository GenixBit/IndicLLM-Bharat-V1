#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Dataset Release Package Builder CLI.

Bundles validated dataset shard manifests, approval records, and SHA-256
audit reports into a sealed dataset release package for training.

Usage:
  python scripts/build_dataset_release.py \
    --manifest data/governed/sangraha_hi/manifest.json \
    --approval data/governed/sangraha_hi/approval.json \
    --output-dir dist/data_releases/sangraha_hi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.release import DatasetReleaseBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a dataset release package from a manifest and approval"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--approval", required=True, help="Path to approval JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for release files")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    approval_path = Path(args.approval)
    output_dir = Path(args.output_dir)

    builder = DatasetReleaseBuilder()

    try:
        release, audit = builder.build(manifest_path, approval_path, output_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        result = {
            "status": "success",
            "release_id": release.release_id,
            "dataset_id": release.dataset_id,
            "shard_count": release.shard_count,
            "records": release.records,
            "package_sha256": release.package_sha256,
            "release_path": str(output_dir / "dataset_release.json"),
            "audit_path": str(output_dir / "audit_report.json"),
        }
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print(f"  🇮🇳 Dataset Release Built: {release.release_id}")
        print("=" * 60)
        print(f"  Dataset:     {release.dataset_id}")
        print(f"  Shards:      {release.shard_count}")
        print(f"  Records:     {release.records}")
        print(f"  SHA-256:     {release.package_sha256}")
        print(f"  Release:     {output_dir / 'dataset_release.json'}")
        print(f"  Audit:       {output_dir / 'audit_report.json'}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
