#!/usr/bin/env python3
"""CLI to generate high-density Indic & Worldwide Knowledge Curriculum.

Usage:
  python scripts/generate_indic_curriculum.py --output-dir data/curriculum --num-samples 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bharat.data.synthetic_curriculum import export_curriculum_datasets  # noqa: E402


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-density Indic + Worldwide pretraining & SFT curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/curriculum",
        help="Directory to save pretrain_corpus.txt and sft_instruct.jsonl",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of curriculum samples to generate",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    print(f"Generating {parsed.num_samples} Indic + Worldwide knowledge samples...")
    pretrain_p, sft_p = export_curriculum_datasets(
        output_dir=parsed.output_dir,
        num_samples=parsed.num_samples,
    )
    print(f"✅ Pretraining corpus saved to: {pretrain_p}")
    print(f"✅ SFT instruction dataset saved to: {sft_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
