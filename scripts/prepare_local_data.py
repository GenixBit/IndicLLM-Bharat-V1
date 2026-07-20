#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.preparation import LocalPreparer, PreparationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local text/JSONL data with governed pipeline",
    )
    parser.add_argument("--input", required=True, help="Path to text/JSONL file or directory")
    parser.add_argument("--source-id", required=True, help="Source identifier (slug)")
    parser.add_argument("--source-version", required=True, help="Source version string")
    parser.add_argument("--license", required=True, help="License identifier (e.g. cc-by-4.0)")
    parser.add_argument("--language", required=True, help="Primary language code (e.g. en, hi)")
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    parser.add_argument("--domain", default="", help="Domain classification")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--max-records-per-shard", type=int, default=10000)
    parser.add_argument("--max-bytes-per-shard", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--blocklist", help="Path to blocklist file for contamination check")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats only, write nothing")
    parser.add_argument("--json", action="store_true", help="Output report as JSON to stdout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"error: input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    config = PreparationConfig(
        source_id=args.source_id,
        source_version=args.source_version,
        license=args.license,
        language=args.language,
        split=args.split,
        domain=args.domain,
        output_dir=args.output_dir,
        max_records_per_shard=args.max_records_per_shard,
        max_bytes_per_shard=args.max_bytes_per_shard,
        dry_run=args.dry_run,
        blocklist_path=args.blocklist,
    )

    preparer = LocalPreparer(config)

    try:
        manifest, report = preparer.prepare(str(input_path))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Total records:       {report.total_records}")
        print(f"Accepted records:    {report.accepted_records}")
        print(f"Rejected records:    {report.rejected_records}")
        print(f"Shards written:      {report.shard_count}")
        if report.rejection_reasons:
            print(f"Rejection reasons:   {dict(sorted(report.rejection_reasons.items()))}")
        if report.language_distribution:
            print(f"Language dist:       {dict(sorted(report.language_distribution.items()))}")
        print(f"Manifest digest:     {report.manifest_digest}")


if __name__ == "__main__":
    main()
