"""CLI tool to ingest Wikipedia knowledge across all 22 Indic Languages and pack into binary shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.wikipedia_ingest import (
    WIKIPEDIA_LANGUAGES,
    ingest_and_pack_wikipedia,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest, Clean, and Pack Wikipedia across 22 Indic Languages + English",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="all",
        help="Comma-separated language codes or 'all' for all 22 Indic languages + English",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/binary_shards",
        help="Directory to save output binary token shards",
    )
    parser.add_argument(
        "--max-docs-per-lang",
        type=int,
        default=500,
        help="Maximum articles per language",
    )
    parser.add_argument(
        "--max-tokens-per-shard",
        type=int,
        default=500_000,
        help="Max tokens per binary shard",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.langs.lower().strip() == "all":
        lang_list = list(WIKIPEDIA_LANGUAGES.keys())
    else:
        lang_list = [
            lang_code.strip().lower() for lang_code in parsed.langs.split(",") if lang_code.strip()
        ]

    print("\n" + "=" * 65)
    print("🌐 IndicLLM-Bharat Wikipedia Ingestion & Sharding Engine")
    print(f"  • Languages Target: {len(lang_list)} ({', '.join(lang_list[:8])}...)")
    print(f"  • Output Directory: {Path(parsed.output_dir).resolve()}")
    print("=" * 65 + "\n")

    result = ingest_and_pack_wikipedia(
        output_dir=parsed.output_dir,
        languages=lang_list,
        max_docs_per_lang=parsed.max_docs_per_lang,
        max_tokens_per_shard=parsed.max_tokens_per_shard,
    )

    print("✅ Ingestion Complete!")
    print(f"  • Articles Ingested:    {result.total_articles}")
    print(f"  • Languages Processed:  {len(result.languages_processed)}")
    print(f"  • Total Characters:     {result.total_characters:,}")
    print(f"  • Binary Shards Built:  {len(result.shards_written)}")
    for s in result.shards_written:
        kb = s.stat().st_size / 1024
        print(f"    - {s.name} ({kb:.1f} KB)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
