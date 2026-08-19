"""CLI tool to generate and export Sovereign Benchmark Leaderboard."""

from __future__ import annotations

import argparse
import sys

from bharat.eval.leaderboard import (
    build_default_sovereign_leaderboard,
    export_leaderboard_files,
    format_markdown_leaderboard,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Cross-Tier Sovereign Benchmark Leaderboard for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/benchmarks",
        help="Directory to save LEADERBOARD.md and leaderboard.json",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    report = build_default_sovereign_leaderboard()

    print("\n" + format_markdown_leaderboard(report))

    exported = export_leaderboard_files(report, parsed.output_dir)
    print("=" * 65)
    print("✅ Exported Leaderboard Files:")
    for k, p in exported.items():
        print(f"  • {k.upper()}: {p.resolve()}")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
