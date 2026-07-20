#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.manifest import DatasetManifest
from bharat.data.sharding import ShardPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan data shards from a dataset manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--max-records", type=int, default=10000, help="Max records per shard")
    parser.add_argument("--max-bytes", type=int, default=0, help="Max bytes per shard")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"error: manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        manifest = DatasetManifest.from_dict(data)
    except Exception as e:
        print(f"error: failed to load manifest: {e}", file=sys.stderr)
        sys.exit(1)

    planner = ShardPlanner(
        dataset_id=manifest.dataset_id,
        split=manifest.split,
        max_records_per_shard=args.max_records,
        max_bytes_per_shard=args.max_bytes,
    )

    try:
        plans = planner.plan(
            total_records=manifest.records,
            total_bytes=manifest.bytes_utf8,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = {
            "dataset_id": manifest.dataset_id,
            "split": manifest.split,
            "total_records": manifest.records,
            "total_bytes": manifest.bytes_utf8,
            "shard_count": len(plans),
            "shards": [p.__dict__ for p in plans],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Shard plan for {manifest.dataset_id} ({manifest.split}):")
        print(f"  Total records: {manifest.records}")
        print(f"  Total bytes:   {manifest.bytes_utf8}")
        print(f"  Shards:        {len(plans)}")
        for p in plans:
            print(
                f"  [{p.index:04d}] {p.shard_id}: "
                f"{p.expected_records} records, {p.expected_bytes} bytes"
            )


if __name__ == "__main__":
    main()
