#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) via TRL.

Usage:
  python train/dpo.py --model checkpoints/gpt2-124m-sft --output checkpoints/gpt2-124m-dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.utils import init_wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("checkpoints/gpt2-124m-sft"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/gpt2-124m-dpo"))
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    init_wandb({"name": "dpo", "wandb": {"enabled": True, "run_name": "dpo", "project": "llm-lab"}})

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_path = str(args.model) if args.model.exists() else "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= args.max_samples:
            break
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if chosen and rejected:
            prompt = chosen.split("Assistant:")[0] + "Assistant:"
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    from datasets import Dataset

    dataset = Dataset.from_list(rows)

    training_args = DPOConfig(
        output_dir=str(args.output),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-7,
        logging_steps=10,
        max_length=512,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"DPO model saved to {args.output}")


if __name__ == "__main__":
    main()
