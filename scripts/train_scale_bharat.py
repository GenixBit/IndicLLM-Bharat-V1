"""CLI tool to execute multi-tier scaling pretraining (1B, 3B, 7B, 10B) on IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.training.scale_trainer import (
    BharatScaleTrainer,
    ScaleTrainerConfig,
    get_scale_tier_config,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 1B to 10B Step-by-Step Sovereign Scale Pretraining for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Target model parameter scaling tier",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of pretraining steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device step",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Context sequence length (tokens)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Compute numerical precision",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="data/binary_shards",
        help="Directory containing memory-mapped binary shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat_scale",
        help="Base directory to save tier checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--dry-run-calc",
        action="store_true",
        help="Calculate parameter count and memory architecture breakdown without training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.dry_run_calc:
        cfg = get_scale_tier_config(parsed.tier)
        print("\n" + "=" * 60)
        print(f"📊 Architecture Spec for Bharat-{parsed.tier.upper()}:")
        print(f"  • Hidden Size:        {cfg.hidden_size}")
        print(f"  • Intermediate Size:  {cfg.intermediate_size}")
        print(f"  • Hidden Layers:      {cfg.num_hidden_layers}")
        print(
            f"  • Attention Heads:    {cfg.num_attention_heads} (Q) / {cfg.num_key_value_heads} (KV)"
        )
        print(f"  • Max Context Window: {cfg.max_position_embeddings}")
        print("=" * 60 + "\n")
        return 0

    config = ScaleTrainerConfig(
        tier=parsed.tier,
        shards_dir=parsed.shards_dir,
        output_dir=parsed.output_dir,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        block_size=parsed.block_size,
        learning_rate=parsed.learning_rate,
        precision=parsed.precision,
        device=parsed.device,
        seed=parsed.seed,
    )

    trainer = BharatScaleTrainer(config)
    result = trainer.train()

    print(
        f"🎯 Pretraining Complete! Tier: {result.tier.upper()} | "
        f"Params: {result.parameter_count:,} | "
        f"Loss: {result.final_loss:.4f} | "
        f"Checkpoint: {Path(result.checkpoint_path).resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
