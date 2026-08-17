from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.preparation import LocalPreparer, PreparationConfig
from bharat.data.processing import ProcessingConfig
from bharat.data.quality import QualityConfig
from bharat.data.sources import load_source_spec

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_DIR = ROOT / "data_registry" / "sources"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Indic data preparation pipeline using governed LocalPreparer"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input text or JSONL file",
    )
    parser.add_argument(
        "--source-id",
        required=True,
        help="Source identifier slug (e.g. indiccorp_v2, sangraha, samanantar, wikipedia_indic)",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Primary Indic language code (e.g. hi, bn, ta, te, mr, gu, kn, ml, pa, or, ur, as, sa)",
    )
    parser.add_argument(
        "--sources-dir",
        default=str(DEFAULT_SOURCES_DIR),
        help="Directory containing data registry source YAML specs",
    )
    parser.add_argument(
        "--output-dir",
        default="data/indic_shards",
        help="Output directory for processed shards and manifest",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--max-records-per-shard",
        type=int,
        default=10000,
        help="Max records per shard file",
    )
    parser.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=64 * 1024 * 1024,
        help="Max uncompressed bytes per shard file",
    )
    parser.add_argument(
        "--blocklist",
        default=None,
        help="Path to blocklist file for contamination filtering",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run filtering without writing shards to disk",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    sources_dir = Path(args.sources_dir)
    source_spec_file = sources_dir / f"{args.source_id}.yaml"

    if source_spec_file.is_file():
        try:
            spec = load_source_spec(source_spec_file)
            source_version = spec.version
            license_id = spec.license
            domain = str(spec.domains[0]) if spec.domains else "general"
        except Exception as e:
            print(f"warning: error loading source spec {source_spec_file}: {e}", file=sys.stderr)
            source_version = "1.0.0"
            license_id = "cc-by-4.0"
            domain = "general"
    else:
        source_version = "1.0.0"
        license_id = "cc-by-4.0"
        domain = "general"

    processing_cfg = ProcessingConfig(
        quality=QualityConfig(min_chars=10, min_words=2, max_chars=100000),
    )

    prep_cfg = PreparationConfig(
        source_id=args.source_id,
        source_version=source_version,
        license=license_id,
        language=args.language,
        split=args.split,
        domain=domain,
        output_dir=args.output_dir,
        max_records_per_shard=args.max_records_per_shard,
        max_bytes_per_shard=args.max_bytes_per_shard,
        blocklist_path=args.blocklist,
        dry_run=args.dry_run,
        processing_config=processing_cfg,
    )

    preparer = LocalPreparer(prep_cfg)
    try:
        _manifest, report = preparer.prepare(input_path)
    except Exception as e:
        print(f"error during Indic data preparation: {e}", file=sys.stderr)
        return 1

    summary = report.to_dict()

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("=" * 60)
        print("🇮🇳 Indic Data Preparation Complete")
        print(f"Source: {args.source_id} (v{source_version}) | Language: {args.language}")
        print(f"Total Records:    {report.total_records:,}")
        print(f"Accepted Records: {report.accepted_records:,}")
        print(f"Rejected Records: {report.rejected_records:,}")
        print(f"Shards Written:   {report.shard_count}")
        if report.manifest_digest:
            print(f"Manifest Digest:  {report.manifest_digest}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
