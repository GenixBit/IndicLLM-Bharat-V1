"""CLI tool to generate and export multilingual preference datasets for IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.preference_curriculum import (
    INDIC_PREFERENCES,
    STEM_AND_GLOBAL_PREFERENCES,
    export_preference_curriculum,
    get_all_preference_samples,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-density multilingual & technical DPO preference curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/preferences/bharat_dpo_curriculum.jsonl",
        help="Path for exported preference JSONL file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed category and language distribution statistics",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    samples = get_all_preference_samples()

    print("🇮🇳 IndicLLM-Bharat Preference Curriculum Generator")
    print(f"  • Indic language pairs: {len(INDIC_PREFERENCES)}")
    print(f"  • STEM & Global pairs:  {len(STEM_AND_GLOBAL_PREFERENCES)}")
    print(f"  • Total preference pairs: {len(samples)}")

    if parsed.stats:
        lang_counts: dict[str, int] = {}
        for s in samples:
            lang_code = s.get("lang", "en")
            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + 1
        print("\nLanguage breakdown:")
        for lang_code, count in sorted(lang_counts.items()):
            print(f"  - {lang_code}: {count} pairs")

    count = export_preference_curriculum(parsed.output)
    print(f"\n✅ Exported {count} preference pairs to: {Path(parsed.output).resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
