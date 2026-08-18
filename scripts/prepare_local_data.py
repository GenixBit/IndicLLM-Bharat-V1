#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Local Data Preparation CLI.

Processes and normalizes raw local text/JSONL datasets through data governance filters,
license validation, deduplication, and manifests generation.

Usage:
  python scripts/prepare_local_data.py \
    --input data/raw/corpus.jsonl \
    --source-id custom_corpus \
    --source-version 1.0.0 \
    --license cc-by-4.0 \
    --language hi \
    --output-dir data/governed/custom_hi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.preparation import LocalPreparer, PreparationConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare local text/JSONL data with governed pipeline",
    )
    parser.add_argument("--input", required=True, help="Path to local text or JSONL file")
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
    parser.add_argument("--created-at", help="ISO-8601 UTC timestamp (e.g. 2026-07-20T12:00:00Z)")

    args = parser.parse_args(argv)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"error: input path not found: {input_path}", file=sys.stderr)
        return 1

    if args.created_at is not None:
        from bharat.data.manifest import _ISO_UTC_RE

        if not _ISO_UTC_RE.match(args.created_at):
            print(
                f"error: invalid --created-at format (expected ISO-8601 UTC): {args.created_at}",
                file=sys.stderr,
            )
            return 1

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
        created_at=args.created_at,
        dry_run=args.dry_run,
        blocklist_path=args.blocklist,
    )

    preparer = LocalPreparer(config=config)

    try:
        manifest, report = preparer.prepare(input_path)
    except Exception as e:
        print(f"error: preparation failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        res = {"dry_run": args.dry_run, **report.to_dict()}
        print(json.dumps(res, indent=2))
    else:
        print("=" * 60)
        print("  🇮🇳 IndicLLM-Bharat — Data Preparation Report")
        print("=" * 60)
        print(f"  Dataset:     {config.source_id}")
        print(f"  Status:      {'DRY RUN' if args.dry_run else 'WRITTEN'}")
        print(f"  Total input: {report.total_records:,} records")
        print(f"  Accepted:    {report.accepted_records:,} records")
        print(f"  Rejected:    {report.rejected_records:,} records")
        print(f"  Shards:      {report.shard_count}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
