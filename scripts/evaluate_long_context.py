"""CLI tool to run long-context (32k) Needle-in-a-Haystack evaluation on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.eval.long_context import LongContextEvaluator


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Long-Context (up to 32k) capabilities on Bharat-1B using YaRN RoPE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model scaling tier to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768],
        help="Context lengths in tokens to test",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[10, 50, 90],
        help="Needle insertion depths in percentages (0-100)",
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
        default="reports/long_context_32k_eval.md",
        help="Target output markdown report path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    print(f"\n📜 Starting Long-Context (32k) Evaluation on Bharat-{parsed.tier.upper()}...")

    evaluator = LongContextEvaluator(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    report = evaluator.run_benchmark(
        context_lengths=parsed.context_lengths,
        depths=parsed.depths,
    )

    print("\n" + report.summary_markdown + "\n")

    if parsed.output_report:
        out_p = Path(parsed.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.summary_markdown, encoding="utf-8")
        print(f"📄 Saved Long-Context Report to: {out_p.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
