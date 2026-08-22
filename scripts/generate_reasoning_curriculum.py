"""CLI tool to generate Sovereign Multilingual Chain-of-Thought (CoT) Reasoning Curriculum."""

from __future__ import annotations

import argparse
import sys

from bharat.reasoning.cot_engine import CoTReasoningEngine


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sovereign Multilingual Chain-of-Thought (CoT) Reasoning Curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reasoning",
        help="Directory to save generated reasoning dataset",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=5,
        help="Dataset size expansion multiplier",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    engine = CoTReasoningEngine()
    curriculum_path = engine.export_curriculum(
        output_dir=parsed.output_dir,
        multiplier=parsed.multiplier,
    )

    print("\n" + "=" * 65)
    print("🧠 IndicLLM-Bharat Sovereign CoT Reasoning Curriculum Generator")
    print(f"  • Output File: {curriculum_path.resolve()} ({curriculum_path.stat().st_size} bytes)")
    print("=" * 65 + "\n")

    print("✅ Generated Reasoning Curriculum Dataset Successfully!")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
