"""CLI tool to evaluate IndicLLM-Bharat on multilingual IndicMMLU and STEM benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.eval.indic_benchmarks import IndicBenchmarkRunner


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 22-language Indic and STEM benchmark evaluation on IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bharat_dpo/final.pt",
        help="Path to model checkpoint to evaluate",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device for inference",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="benchmark_report.md",
        help="Path to export markdown evaluation report",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to export raw JSON metrics",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    print(f"📊 Initializing IndicLLM Benchmark Runner for checkpoint: {parsed.checkpoint}...")

    runner = IndicBenchmarkRunner(
        checkpoint_path=parsed.checkpoint,
        tokenizer_name=parsed.tokenizer,
        device=parsed.device,
    )

    report = runner.generate_report()

    print("\n" + report.summary_markdown + "\n")

    if parsed.output_report:
        out_p = Path(parsed.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.summary_markdown, encoding="utf-8")
        print(f"📄 Saved Markdown Report to: {out_p.resolve()}")

    if parsed.json_output:
        json_p = Path(parsed.json_output)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        json_p.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"📈 Saved JSON Metrics to: {json_p.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
