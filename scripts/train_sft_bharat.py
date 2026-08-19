"""CLI tool to execute Supervised Fine-Tuning (SFT) on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.posttraining.sft_trainer import (
    BharatSFTTrainer,
    SFTTrainingConfig,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sovereign SFT Alignment on IndicLLM-Bharat",
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
        "--checkpoint",
        type=str,
        default=None,
        help="Optional base pretrained checkpoint path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sft/bharat_instruction_curriculum.jsonl",
        help="Path to SFT JSONL instruction dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat_sft",
        help="Directory to save SFT checkpoints",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Total SFT training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-step batch size",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Peak SFT learning rate",
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

    config = SFTTrainingConfig(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        data_path=parsed.data_path,
        output_dir=parsed.output_dir,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        block_size=parsed.block_size,
        learning_rate=parsed.learning_rate,
        device=parsed.device,
        seed=parsed.seed,
    )

    trainer = BharatSFTTrainer(config)
    result = trainer.train()

    print(
        f"🎯 SFT Complete! Tier: {result.tier.upper()} | "
        f"Loss: {result.final_loss:.4f} | "
        f"Active Tokens: {result.active_tokens:,} | "
        f"Saved Checkpoint: {Path(result.checkpoint_path).resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
