#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.release import DatasetReleaseBuilder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a dataset release package from a manifest and approval"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--approval", required=True, help="Path to approval JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for release files")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    approval_path = Path(args.approval)
    output_dir = Path(args.output_dir)

    builder = DatasetReleaseBuilder()

    try:
        release, audit = builder.build(manifest_path, approval_path, output_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

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
        print(json.dumps(result))
    else:
        print(f"Release built: {release.release_id}")
        print(f"  Dataset:     {release.dataset_id}")
        print(f"  Shards:      {release.shard_count}")
        print(f"  Records:     {release.records}")
        print(f"  SHA-256:     {release.package_sha256}")
        print(f"  Release:     {output_dir / 'dataset_release.json'}")
        print(f"  Audit:       {output_dir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
