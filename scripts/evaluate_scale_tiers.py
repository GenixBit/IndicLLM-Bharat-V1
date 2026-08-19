"""CLI tool to evaluate and benchmark all model scaling tiers (1B, 3B, 7B, 10B)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.eval.scale_evaluator import ScaleTierEvaluator


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-tier scaling evaluation across 1B, 3B, 7B, and 10B tiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["1b", "3b", "7b", "10b"],
        help="List of model tiers to evaluate",
    )
    parser.add_argument(
        "--checkpoints-base",
        type=str,
        default="checkpoints/bharat_scale",
        help="Base directory containing tier checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda", "auto"],
        help="Compute device",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/scaling_tiers_matrix.md",
        help="Target output markdown report path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    print(f"📊 Running multi-tier scaling evaluation across: {parsed.tiers}...")

    evaluator = ScaleTierEvaluator(
        tiers=parsed.tiers,
        checkpoints_base=parsed.checkpoints_base,
        device=parsed.device,
    )

    report = evaluator.generate_comparison_matrix()
    print("\n" + report.summary_markdown + "\n")

    if parsed.output_report:
        out_p = Path(parsed.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.summary_markdown, encoding="utf-8")
        print(f"📄 Saved Scaling Matrix Report to: {out_p.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
