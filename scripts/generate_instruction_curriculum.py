"""CLI tool to generate and export Multilingual SFT Instruction Curriculum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.data.instruction_curriculum import (
    INDIC_INSTRUCTION_DATA,
    STEM_INSTRUCTION_DATA,
    export_instruction_curriculum,
    get_all_instruction_curriculum,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate High-Density SFT Instruction Curriculum for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/bharat_instruction_curriculum.jsonl",
        help="Target output JSONL path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    items = get_all_instruction_curriculum()

    print("\n" + "=" * 65)
    print("🎓 IndicLLM-Bharat Multilingual SFT Instruction Curriculum")
    print(f"  • STEM & Scientific Dialogues:   {len(STEM_INSTRUCTION_DATA)}")
    print(f"  • 22 Indian Language Dialogues:  {len(INDIC_INSTRUCTION_DATA)}")
    print(f"  • Total Instruction-Answer Pairs:{len(items)}")
    print("=" * 65 + "\n")

    count = export_instruction_curriculum(parsed.output)
    print(f"✅ Exported {count} instruction pairs to: {Path(parsed.output).resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
