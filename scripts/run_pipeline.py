from __future__ import annotations

import argparse
import json
import sys

from bharat.training.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the full IndicLLM-Bharat training lifecycle: Pretrain -> SFT -> DPO -> Eval"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline YAML configuration (e.g. configs/pipeline/bharat-350m-e2e.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override execution device across all stages (e.g. cpu, cuda, mps)",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use synthetic token stream for pretraining smoke testing",
    )
    parser.add_argument(
        "--pretrain-iters",
        type=int,
        default=None,
        help="Override pretraining max iterations",
    )
    parser.add_argument(
        "--sft-iters",
        type=int,
        default=None,
        help="Override SFT max iterations",
    )
    parser.add_argument(
        "--dpo-iters",
        type=int,
        default=None,
        help="Override DPO max iterations",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining stage",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT stage",
    )
    parser.add_argument(
        "--skip-dpo",
        action="store_true",
        help="Skip DPO stage",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and stage dependencies without executing compute",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output final pipeline results as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"error loading pipeline config '{args.config}': {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.device:
        config.pretrain.device = args.device
        config.sft.device = args.device
        config.dpo.device = args.device
        config.eval.device = args.device

    if args.synthetic_data:
        config.pretrain.synthetic_data = True

    if args.pretrain_iters is not None:
        config.pretrain.max_iters = args.pretrain_iters

    if args.sft_iters is not None:
        config.sft.max_iters = args.sft_iters

    if args.dpo_iters is not None:
        config.dpo.max_iters = args.dpo_iters

    if args.skip_pretrain:
        config.pretrain.enabled = False

    if args.skip_sft:
        config.sft.enabled = False

    if args.skip_dpo:
        config.dpo.enabled = False

    if args.skip_eval:
        config.eval.enabled = False

    if args.dry_run:
        summary = {
            "dry_run": True,
            "pipeline_name": config.name,
            "output_dir": config.output_dir,
            "stages": {
                "pretrain": config.pretrain.enabled,
                "sft": config.sft.enabled,
                "dpo": config.dpo.enabled,
                "eval": config.eval.enabled,
            },
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("=" * 60)
            print("🔍 Pipeline Configuration Validated (Dry Run)")
            print(f"Name:       {config.name}")
            print(f"Output Dir: {config.output_dir}")
            print(f"Stages:     {summary['stages']}")
            print("=" * 60)
        return 0

    if not args.json:
        print("=" * 60)
        print(f"🚀 Launching Bharat Training Pipeline: {config.name}")
        print(f"Target Output Directory: {config.output_dir}")
        print("=" * 60)

    try:
        result = run_pipeline(config)
    except Exception as e:
        print(f"error during pipeline execution: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("=" * 60)
        print("🏁 Pipeline Completed Successfully!")
        print(f"Completed Stages: {', '.join(result.completed_stages)}")
        print(f"Total Duration:   {result.total_duration_sec:.2f}s")
        if result.final_checkpoint:
            print(f"Final Checkpoint: {result.final_checkpoint}")
        if result.eval_scores:
            print(f"Eval Scores:      {result.eval_scores}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
