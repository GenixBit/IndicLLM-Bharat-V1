"""CLI tool to evaluate Step-by-Step Chain-of-Thought (CoT) Reasoning on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys

from bharat.reasoning.verifier import ReasoningVerifier


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Multilingual Chain-of-Thought Reasoning on IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model tier",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/benchmarks",
        help="Directory to save reasoning benchmark reports",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    verifier = ReasoningVerifier(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    report = verifier.evaluate_problems()
    md_p, json_p = verifier.export_reports(report, output_dir=parsed.output_dir)

    print("\n" + "=" * 65)
    print("🧠 IndicLLM-Bharat Multilingual CoT Reasoning Audit")
    print(f"  • Model Tier:           {report.model_tier.upper()}")
    print(
        f"  • Structure Compliance: {report.structure_valid_pct:.1f}% ({report.structure_valid_count}/{report.total_problems})"
    )
    print("=" * 65 + "\n")

    print("📊 Domain Breakdown:")
    for dom, score in report.per_domain_validity.items():
        print(f"  • {dom:30s}: {score:.1f}%")

    print("\n✅ Exported Benchmark Reports:")
    print(f"  • Markdown Report: {md_p}")
    print(f"  • JSON Audit File: {json_p}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
