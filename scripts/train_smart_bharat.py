#!/usr/bin/env python3
"""CLI to execute multi-tier native smart training for IndicLLM-Bharat.

Usage:
  python scripts/train_smart_bharat.py --model-size small --pretrain-iters 100 --sft-iters 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bharat.training.smart_trainer import SmartTrainerConfig, train_smart_bharat  # noqa: E402


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IndicLLM-Bharat natively on Indic + Worldwide Knowledge Curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        default="small",
        help="Target model scale tier",
    )
    parser.add_argument(
        "--curriculum-dir",
        type=str,
        default="data/curriculum",
        help="Directory containing curriculum datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat_smart",
        help="Directory to save final model checkpoint",
    )
    parser.add_argument(
        "--pretrain-iters",
        type=int,
        default=100,
        help="Number of pretraining iterations",
    )
    parser.add_argument(
        "--sft-iters",
        type=int,
        default=50,
        help="Number of SFT instruction tuning iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    cfg = SmartTrainerConfig(
        model_tier=parsed.model_size,
        curriculum_dir=parsed.curriculum_dir,
        output_dir=parsed.output_dir,
        pretrain_iters=parsed.pretrain_iters,
        sft_iters=parsed.sft_iters,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        device=parsed.device,
    )
    res = train_smart_bharat(cfg)
    print(f"\n🎉 Bharat-{res.model_tier.upper()} training complete!")
    print(f"   Final Pretrain Loss: {res.final_pretrain_loss:.4f}")
    print(f"   Final SFT Loss:      {res.final_sft_loss:.4f}")
    print(f"   Checkpoint:          {res.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
