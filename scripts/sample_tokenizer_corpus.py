from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.tokenizer.sampler import (
    SamplerConfig,
    sample_tokenizer_corpus,
)


def _parse_caps(values: list[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not values:
        return result
    for item in values:
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid cap format: {item!r}. Expected key:count")
        key, val_str = parts
        try:
            val = int(val_str)
        except ValueError:
            raise ValueError(f"Invalid cap value for {key!r}: {val_str!r} must be an integer")
        if val <= 0:
            raise ValueError(f"Cap for {key!r} must be positive, got {val}")
        result[key] = val
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic tokenizer corpus sampler from approved dataset releases"
    )
    parser.add_argument(
        "--release-root",
        action="append",
        required=True,
        dest="release_roots",
        help="Release root directory (repeatable)",
    )
    parser.add_argument(
        "--manifest-path",
        action="append",
        required=True,
        dest="manifest_paths",
        help="Manifest JSON path, one per release root (repeatable)",
    )
    parser.add_argument(
        "--approval-path",
        action="append",
        required=True,
        dest="approval_paths",
        help="Approval JSON path, one per release root (repeatable)",
    )
    parser.add_argument("--version", required=True, help="Sampler version string")
    parser.add_argument("--seed", required=True, type=int, help="Deterministic selection seed")
    parser.add_argument(
        "--max-total-records",
        type=int,
        default=0,
        help="Maximum total records (0 = unlimited)",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=0,
        help="Maximum total UTF-8 bytes (0 = unlimited)",
    )
    parser.add_argument(
        "--max-records-per-source",
        nargs="*",
        default=[],
        help="source_id:count pairs",
    )
    parser.add_argument(
        "--max-bytes-per-source",
        nargs="*",
        default=[],
        help="source_id:byte_count pairs",
    )
    parser.add_argument(
        "--max-records-per-language",
        nargs="*",
        default=[],
        help="lang:count pairs",
    )
    parser.add_argument(
        "--max-bytes-per-language",
        nargs="*",
        default=[],
        help="lang:byte_count pairs",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name for record text (default: text)",
    )
    parser.add_argument(
        "--language-field",
        default="language",
        help="Field name for language (default: language)",
    )
    parser.add_argument(
        "--domain-field",
        default="domain",
        help="Field name for domain (default: domain)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_false",
        dest="exact_dedup",
        help="Disable exact content deduplication",
    )
    parser.add_argument("--output-corpus", default="", help="Corpus JSONL output path")
    parser.add_argument("--output-manifest", default="", help="Manifest JSON output path")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write output (default is dry-run)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)

    rroots: list[str] = args.release_roots
    mpaths: list[str] = args.manifest_paths
    apaths: list[str] = args.approval_paths

    if not (len(rroots) == len(mpaths) == len(apaths)):
        print(
            f"Error: --release-root ({len(rroots)}), --manifest-path ({len(mpaths)}), "
            f"and --approval-path ({len(apaths)}) counts must match",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.execute:
        if not args.output_corpus:
            print("Error: --output-corpus required when --execute is set", file=sys.stderr)
            sys.exit(1)
        if not args.output_manifest:
            print("Error: --output-manifest required when --execute is set", file=sys.stderr)
            sys.exit(1)
    elif args.output_corpus or args.output_manifest:
        print(
            "Error: --execute is required when --output-corpus or --output-manifest is set",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        max_records_per_source = _parse_caps(args.max_records_per_source)
        max_bytes_per_source = _parse_caps(args.max_bytes_per_source)
        max_records_per_language = _parse_caps(args.max_records_per_language)
        max_bytes_per_language = _parse_caps(args.max_bytes_per_language)
    except ValueError as e:
        print(f"Error parsing caps: {e}", file=sys.stderr)
        sys.exit(1)

    config = SamplerConfig(
        version=args.version,
        seed=args.seed,
        max_total_records=args.max_total_records,
        max_total_bytes=args.max_total_bytes,
        max_records_per_source=max_records_per_source,
        max_bytes_per_source=max_bytes_per_source,
        max_records_per_language=max_records_per_language,
        max_bytes_per_language=max_bytes_per_language,
        text_field=args.text_field,
        language_field=args.language_field,
        domain_field=args.domain_field,
        exact_dedup=args.exact_dedup,
        output_corpus=args.output_corpus,
        output_manifest=args.output_manifest,
    )

    try:
        manifest = sample_tokenizer_corpus(
            release_roots=[Path(p) for p in rroots],
            manifest_paths=[Path(p) for p in mpaths],
            approval_paths=[Path(p) for p in apaths],
            config=config,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "status": "dry-run" if config.dry_run else "success",
        "total_candidates": manifest.total_candidates,
        "total_selected": manifest.total_selected,
        "exact_dedup_removed": manifest.exact_dedup_removed,
        "per_source_cap_removed": manifest.per_source_cap_removed,
        "per_language_cap_removed": manifest.per_language_cap_removed,
        "global_cap_removed": manifest.global_cap_removed,
        "total_corpus_bytes": manifest.total_corpus_bytes,
        "corpus_sha256": manifest.corpus_sha256,
    }

    if not config.dry_run:
        result["manifest_sha256"] = manifest.manifest_sha256

    print(json.dumps(result, indent=2))

    if config.dry_run:
        print("Dry-run complete. Pass --execute to produce output.", file=sys.stderr)


if __name__ == "__main__":
    main()
