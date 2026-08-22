"""CLI tool to execute the Sovereign Safety Guardrails Audit on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys

from bharat.safety.guardrails import SovereignSafetyGuardrails


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sovereign Safety Guardrails & Alignment Evaluation for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model architecture tier",
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
        default="docs/safety",
        help="Directory to save safety report and audit JSON",
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

    evaluator = SovereignSafetyGuardrails(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    report = evaluator.run_safety_audit()
    md_p, json_p = evaluator.export_audit_reports(report, output_dir=parsed.output_dir)

    print("\n" + "=" * 65)
    print("🛡️ IndicLLM-Bharat Sovereign Safety & Guardrails Audit")
    print(f"  • Model Tier:       {report.model_tier.upper()}")
    print(
        f"  • Safety Pass Rate: {report.safety_pass_rate_pct:.1f}% ({report.passed_tests}/{report.total_tests})"
    )
    print("=" * 65 + "\n")

    print("📊 Category Pass Rates:")
    for cat, score in report.category_breakdown.items():
        print(f"  • {cat:25s}: {score:.1f}%")

    print("\n✅ Exported Safety Reports:")
    print(f"  • Markdown Report: {md_p}")
    print(f"  • JSON Audit File: {json_p}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
