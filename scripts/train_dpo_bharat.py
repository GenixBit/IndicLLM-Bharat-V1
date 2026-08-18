"""CLI tool to run sovereign Direct Preference Optimization (DPO) on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.training.dpo_trainer import BharatDPOTrainer, DPOTrainerConfig


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sovereign Direct Preference Optimization (DPO) on IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default="checkpoints/bharat_smart/final.pt",
        help="Path to starting SFT checkpoint",
    )
    parser.add_argument(
        "--preference-data",
        type=str,
        default="data/preferences/bharat_dpo_curriculum.jsonl",
        help="Path to preference dataset JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat_dpo",
        help="Directory to save aligned DPO checkpoints",
    )
    parser.add_argument(
        "--model-tier",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        default="small",
        help="Model architecture tier",
    )
    parser.add_argument("--max-iters", type=int, default=60, help="Number of DPO training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Compute device"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    config = DPOTrainerConfig(
        sft_checkpoint=parsed.sft_checkpoint,
        preference_data=parsed.preference_data,
        output_dir=parsed.output_dir,
        model_tier=parsed.model_tier,
        max_iters=parsed.max_iters,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        beta=parsed.beta,
        device=parsed.device,
        seed=parsed.seed,
    )

    trainer = BharatDPOTrainer(config)
    result = trainer.train()

    print(
        f"🎯 Alignment completed! Final Loss: {result.final_loss:.4f} | "
        f"Reward Accuracy: {result.final_reward_accuracy*100:.1f}% | "
        f"Checkpoint: {Path(result.checkpoint_path).resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
