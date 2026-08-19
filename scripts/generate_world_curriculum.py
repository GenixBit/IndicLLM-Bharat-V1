"""CLI tool to generate and pack World & Indic Knowledge Pretraining Datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.world_knowledge import (
    CS_AND_AI_DATA,
    GLOBAL_SCIENCE_DATA,
    INDIC_22_LANGUAGES_DATA,
    WORLD_GEOGRAPHY_DATA,
    WORLD_HISTORY_DATA,
    export_world_knowledge_corpus,
    get_all_world_knowledge_documents,
    pack_world_knowledge_shards,
)
from bharat.tokenizer import load_tokenizer


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and pack World & Indic Knowledge Pretraining Datasets for 1B to 10B scaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/world_knowledge/world_corpus.jsonl",
        help="Target output JSONL path",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Directly pack into high-speed binary memory-mapped shards",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="data/binary_shards",
        help="Directory to store binary shards if --pack is enabled",
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
    docs = get_all_world_knowledge_documents()

    print("\n" + "=" * 60)
    print("🌍 IndicLLM-Bharat World & Multilingual Knowledge Corpus")
    print(f"  • Global Science & STEM:    {len(GLOBAL_SCIENCE_DATA)} domains")
    print(f"  • World History & Civs:     {len(WORLD_HISTORY_DATA)} domains")
    print(f"  • World Geography:          {len(WORLD_GEOGRAPHY_DATA)} domains")
    print(f"  • Computer Science & AI:    {len(CS_AND_AI_DATA)} domains")
    print(f"  • 22 Scheduled Languages:   {len(INDIC_22_LANGUAGES_DATA)} languages")
    print(f"  • Total Knowledge Modules:  {len(docs)}")
    print("=" * 60 + "\n")

    count = export_world_knowledge_corpus(parsed.output)
    print(f"✅ Exported {count} knowledge documents to: {Path(parsed.output).resolve()}")

    if parsed.pack:
        print(f"\n📦 Packing into binary token shards using '{parsed.tokenizer}' tokenizer...")
        tokenizer = load_tokenizer(parsed.tokenizer)
        shards = pack_world_knowledge_shards(tokenizer, parsed.shards_dir)
        print(f"✅ Created {len(shards)} binary shards in {Path(parsed.shards_dir).resolve()}:")
        for s in shards:
            size_kb = s.stat().st_size / 1024
            print(f"  • {s.name} ({size_kb:.1f} KB)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
