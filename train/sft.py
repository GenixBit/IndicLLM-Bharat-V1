#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — SFT (Supervised Fine-Tuning)

Fine-tunes a pretrained checkpoint on instruction/response pairs.

Usage:
  python train/sft.py \
    --base-checkpoint checkpoints/gpt2-124m/ckpt.pt \
    --data data/sft/train.jsonl \
    --config configs/gpt2-124m.yaml \
    --output checkpoints/gpt2-124m-sft

Data format (JSONL):
  {"instruction": "...", "response": "..."}
  {"instruction": "...", "response": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT, GPTConfig
from train.utils import get_device_preference, init_wandb, load_config

PROMPT_TEMPLATE = "<|instruction|>{instruction}<|response|>{response}<|endoftext|>"


# ── Dataset ──────────────────────────────────────────────────
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.samples: list[list[int]] = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                text = PROMPT_TEMPLATE.format(
                    instruction=item.get("instruction", ""),
                    response=item.get("response", item.get("output", "")),
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1:
                    self.samples.append(ids)
        print(f"  Loaded {len(self.samples)} SFT samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx][: self.block_size + 1]
        # Pad if shorter than block_size
        if len(ids) < self.block_size + 1:
            ids = ids + [0] * (self.block_size + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_special_tokens({
        "additional_special_tokens": ["<|instruction|>", "<|response|>"]
    })
    return tok


# ── LR schedule ──────────────────────────────────────────────
def get_lr(it: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if it < warmup:
        return max_lr * it / warmup
    if it > total:
        return min_lr
    ratio = (it - warmup) / (total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IndicLLM SFT fine-tuning")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data",            type=Path, required=True)
    parser.add_argument("--config",          type=Path, default=Path("configs/gpt2-124m.yaml"))
    parser.add_argument("--output",          type=Path, default=Path("checkpoints/gpt2-124m-sft"))
    parser.add_argument("--max-iters",       type=int,  default=5000)
    parser.add_argument("--batch-size",      type=int,  default=8)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--warmup-iters",    type=int,  default=200)
    args = parser.parse_args()

    device = get_device_preference()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    print(f"\n{'='*60}")
    print(f"  IndicLLM SFT Fine-tuning")
    print(f"  Base : {args.base_checkpoint}")
    print(f"  Data : {args.data}")
    print(f"  Out  : {args.output}")
    print(f"  Device: {device.upper()}")
    print(f"{'='*60}\n")

    # Load base model
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if "model" in saved_cfg:
        model_cfg = saved_cfg["model"]
    model = GPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded base model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    tokenizer = get_tokenizer()
    dataset = SFTDataset(args.data, tokenizer, model_cfg["block_size"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, drop_last=True)

    # Only fine-tune attention + MLP (freeze embeddings for stability)
    for name, param in model.named_parameters():
        if "wte" in name or "wpe" in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable/1e6:.1f}M (embeddings frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    ctx = nullcontext() if device in ("cpu", "mps") \
          else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    args.output.mkdir(parents=True, exist_ok=True)
    init_wandb({**cfg, "wandb": {**cfg.get("wandb", {}), "run_name": "sft"}})

    model.train()
    step = 0
    best_loss = float("inf")
    for epoch in range(100):
        for x, y in loader:
            if step >= args.max_iters:
                break
            lr = get_lr(step, args.warmup_iters, args.max_iters, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                print(f"  step {step:>5}: loss {loss.item():.4f}  lr {lr:.2e}")

            if step % 500 == 0 and step > 0:
                ckpt_out = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                }
                torch.save(ckpt_out, args.output / "ckpt.pt")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt_out, args.output / "best.pt")
                print(f"  Checkpoint saved → {args.output}/ckpt.pt")
            step += 1
        if step >= args.max_iters:
            break

    torch.save({"model": model.state_dict(), "config": cfg}, args.output / "final.pt")
    print(f"\n  SFT complete. Best loss: {best_loss:.4f}")
    print(f"  Final model → {args.output}/final.pt")


if __name__ == "__main__":
    main()
