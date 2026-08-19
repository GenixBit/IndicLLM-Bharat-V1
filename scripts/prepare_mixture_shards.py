"""CLI tool to prepare, filter, and pack governed data mixtures into binary shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.mixture import (
    MixtureWeights,
    stream_and_pack_mixture,
)
from bharat.tokenizer import load_tokenizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and pack governed Indic + World Data Mixtures into Binary Shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/binary_shards",
        help="Directory to save packed binary shards",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--max-tokens-per-shard",
        type=int,
        default=500_000,
        help="Maximum tokens per binary shard",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on documents to process",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    weights = MixtureWeights()
    weights.validate()

    print("\n" + "=" * 65)
    print("📦 IndicLLM-Bharat Governed Data Mixture Pipeline")
    print(f"  • Indic Multilingual Web:  {weights.indic_multilingual * 100:.0f}%")
    print(f"  • STEM / Science / Math:   {weights.stem_science_math * 100:.0f}%")
    print(f"  • World Knowledge / Civics:{weights.world_knowledge_history * 100:.0f}%")
    print(f"  • Code & Algorithms:       {weights.code_algorithms * 100:.0f}%")
    print(f"  • Output Directory:        {Path(parsed.output_dir).resolve()}")
    print("=" * 65 + "\n")

    tokenizer = load_tokenizer(parsed.tokenizer)
    shards = stream_and_pack_mixture(
        tokenizer=tokenizer,
        output_dir=parsed.output_dir,
        max_tokens_per_shard=parsed.max_tokens_per_shard,
        max_docs=parsed.max_docs,
    )

    print(f"✅ Generated {len(shards)} binary shards:")
    for s in shards:
        size_kb = s.stat().st_size / 1024
        print(f"  • {s.name} ({size_kb:.1f} KB)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
