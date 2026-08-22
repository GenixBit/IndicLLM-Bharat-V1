"""CLI tool to generate sovereign Constitutional AI synthetic alignment data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.synthetic.constitutional import ConstitutionalDataEngine


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sovereign Constitutional AI Synthetic Datasets for DPO and SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/constitutional",
        help="Directory to save generated datasets",
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

    engine = ConstitutionalDataEngine()
    dpo_p, sft_p = engine.export_datasets(
        output_dir=parsed.output_dir,
        num_multiplier=parsed.multiplier,
    )

    print("\n" + "=" * 65)
    print("🇮🇳 Sovereign Constitutional AI Synthetic Data Generator")
    print(f"  • Output Directory: {Path(parsed.output_dir).resolve()}")
    print("=" * 65 + "\n")

    print("✅ Generated Datasets:")
    print(f"  • DPO Preferences:  {dpo_p} ({dpo_p.stat().st_size} bytes)")
    print(f"  • SFT Instructions: {sft_p} ({sft_p.stat().st_size} bytes)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
