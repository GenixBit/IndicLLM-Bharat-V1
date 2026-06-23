#!/usr/bin/env python3
"""
Supervised fine-tuning on instruction datasets (OpenHermes-style).

Usage:
  python train/sft.py --base-checkpoint checkpoints/gpt2-124m/final.pt --output checkpoints/gpt2-124m-sft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.utils import init_wandb


def format_example(row: dict) -> str:
    instruction = row.get("instruction") or row.get("prompt") or ""
    inp = row.get("input") or ""
    output = row.get("output") or row.get("response") or row.get("content") or ""
    if inp:
        instruction = f"{instruction}\n{inp}".strip()
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt2-124m/final.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/gpt2-124m-sft"))
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    init_wandb({"name": "sft", "wandb": {"enabled": True, "run_name": "sft", "project": "llm-lab"}})

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if args.base_checkpoint.exists():
        ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            try:
                model.load_state_dict(ckpt["model"], strict=False)
                print(f"Loaded weights from {args.base_checkpoint}")
            except Exception as e:
                print(f"Could not load checkpoint strictly: {e}; using GPT-2 base.")

    ds = load_dataset(args.dataset, split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= args.max_samples:
            break
        conv = row.get("conversations")
        if conv:
            parts = [f"{m.get('from', 'user')}: {m.get('value', '')}" for m in conv]
            texts.append("\n".join(parts))
        else:
            texts.append(format_example(row))

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"epoch {epoch} loss {loss.item():.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"SFT model saved to {args.output}")


if __name__ == "__main__":
    main()
