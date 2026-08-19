"""CLI tool to execute the Unified Sovereign Training Pipeline on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys

from bharat.pipeline.orchestrator import (
    PipelineConfig,
    SovereignPipelineOrchestrator,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run End-to-End Sovereign Pretraining, Alignment, and Eval Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Target model tier",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages: data,pretrain,sft,dpo,export,eval or 'all'",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="workspace/pipeline_run",
        help="Workspace directory for pipeline artifacts",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=50,
        help="Optimizer steps for pretraining",
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=30,
        help="Optimizer steps for SFT alignment",
    )
    parser.add_argument(
        "--dpo-steps",
        type=int,
        default=20,
        help="Optimizer steps for DPO preference tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per training step",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.stages.lower().strip() == "all":
        stage_list = ["data", "pretrain", "sft", "dpo", "export", "eval"]
    else:
        stage_list = [s.strip() for s in parsed.stages.split(",") if s.strip()]

    config = PipelineConfig(
        tier=parsed.tier,
        work_dir=parsed.work_dir,
        stages=stage_list,
        pretrain_steps=parsed.pretrain_steps,
        sft_steps=parsed.sft_steps,
        dpo_steps=parsed.dpo_steps,
        batch_size=parsed.batch_size,
        device=parsed.device,
        seed=parsed.seed,
    )

    orchestrator = SovereignPipelineOrchestrator(config)
    manifest = orchestrator.run_pipeline()

    print(
        f"🎯 Complete! Pipeline: {manifest.pipeline_id} | "
        f"Tier: {manifest.tier.upper()} | "
        f"Duration: {manifest.total_duration_seconds:.2f}s | "
        f"SHA-256: {manifest.manifest_sha256[:12]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
