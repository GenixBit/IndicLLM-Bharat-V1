"""CLI tool to pack text/JSONL datasets into memory-mapped binary shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.binary_stream import pack_text_corpus
from bharat.tokenizer import load_tokenizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack text corpora into high-throughput memory-mapped binary shards for 10B training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text or JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/binary_shards",
        help="Target directory for binary shards",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="bharat_shard",
        help="Filename prefix for shards",
    )
    parser.add_argument(
        "--max-tokens-per-shard",
        type=int,
        default=5_000_000,
        help="Maximum number of tokens per shard file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name or path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    input_path = Path(parsed.input)
    if not input_path.is_file():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    print(f"📦 Loading tokenizer: {parsed.tokenizer}...")
    tokenizer = load_tokenizer(parsed.tokenizer)

    print(f"🚀 Packing tokens from {input_path} into {parsed.output_dir}...")
    shards = pack_text_corpus(
        tokenizer=tokenizer,
        input_file=input_path,
        output_dir=parsed.output_dir,
        prefix=parsed.prefix,
        max_tokens_per_shard=parsed.max_tokens_per_shard,
    )

    print(f"\n✅ Finished packing {len(shards)} binary shards:")
    for s in shards:
        size_mb = s.stat().st_size / (1024 * 1024)
        print(f"  • {s.name} ({size_mb:.2f} MB)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
