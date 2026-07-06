#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — SFT (Supervised Fine-Tuning)

Fine-tunes a pretrained checkpoint on instruction/response pairs
with assistant-only loss masking.

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
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.tokenizer import load_tokenizer
from bharat.tokenizer.metadata import tokenizer_hash
from bharat.training.checkpointing import get_git_sha, get_package_versions
from train.pretrain import GPT, GPTConfig
from train.utils import get_device_preference, init_wandb, load_config

PROMPT_TEMPLATE = "<|instruction|>{instruction}<|response|>{response}<|endoftext|>"
RESPONSE_SEPARATOR = "<|response|>"


# ── Dataset ──────────────────────────────────────────────────
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int, pad_token_id: int = 50256):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.samples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                instruction = item.get("instruction", "")
                response = item.get("response", item.get("output", ""))
                full_text = PROMPT_TEMPLATE.format(instruction=instruction, response=response)
                response_start = full_text.index(RESPONSE_SEPARATOR) + len(RESPONSE_SEPARATOR)
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                if not full_ids:
                    continue
                # Find response token boundary
                prefix = full_text[:response_start]
                prompt_end = len(tokenizer.encode(prefix, add_special_tokens=False))
                if prompt_end >= len(full_ids):
                    continue
                self.samples.append(
                    {
                        "ids": full_ids,
                        "prompt_end": prompt_end,
                    }
                )
        print(f"  Loaded {len(self.samples)} SFT samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ids = item["ids"][: self.block_size + 1]
        prompt_end = item["prompt_end"]
        if prompt_end > self.block_size:
            prompt_end = self.block_size
        # Pad if shorter than block_size + 1
        if len(ids) < self.block_size + 1:
            ids = ids + [self.pad_token_id] * (self.block_size + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # Mask non-assistant tokens with -100 in labels
        y[:prompt_end] = -100
        return x, y


def get_tokenizer(tok_src: str | None = None):
    tokenizer = load_tokenizer(tok_src)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|instruction|>", "<|response|>"]})
    return tokenizer


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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/gpt2-124m.yaml"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/gpt2-124m-sft"))
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    args = parser.parse_args()

    device = get_device_preference()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    print(f"\n{'=' * 60}")
    print("  IndicLLM SFT Fine-tuning")
    print(f"  Base : {args.base_checkpoint}")
    print(f"  Data : {args.data}")
    print(f"  Out  : {args.output}")
    print(f"  Device: {device.upper()}")
    print(f"{'=' * 60}\n")

    # Load base model
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if "model" in saved_cfg:
        model_cfg = saved_cfg["model"]
    model = GPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded base model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Tokenizer: use config tokenizer source or default to GPT-2
    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = get_tokenizer(tok_src)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|instruction|>", "<|response|>"]}
    )
    print(
        f"  Tokenizer: {tokenizer.tokenizer_type} (vocab={tokenizer.vocab_size}, added={num_added} tokens)"
    )

    # Resize model embeddings if special tokens were added
    if num_added > 0:
        old_vocab = model_cfg.get("vocab_size", 50257)
        new_vocab = old_vocab + num_added
        model.transformer.wte = torch.nn.Embedding(new_vocab, model_cfg["n_embd"]).to(device)
        model.lm_head = torch.nn.Linear(model_cfg["n_embd"], new_vocab, bias=False).to(device)
        model.transformer.wte.weight = model.lm_head.weight
        model_cfg["vocab_size"] = new_vocab
        print(f"  Embedding resized: {old_vocab} → {new_vocab}")

    dataset = SFTDataset(
        args.data, tokenizer, model_cfg["block_size"], pad_token_id=tokenizer.pad_token_id
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Only fine-tune attention + MLP (freeze embeddings for stability)
    for name, param in model.named_parameters():
        if "wte" in name or "wpe" in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable / 1e6:.1f}M (embeddings frozen)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    ctx = (
        nullcontext()
        if device in ("cpu", "mps")
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

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
                    "metadata": {
                        "tokenizer_type": tokenizer.tokenizer_type,
                        "tokenizer_hash": tokenizer_hash(tokenizer),
                        "vocab_size": tokenizer.vocab_size,
                        "git_sha": get_git_sha(),
                        "training_step": step,
                        "config_name": cfg.get("name", ""),
                        "package_versions": get_package_versions(),
                    },
                }
                torch.save(ckpt_out, args.output / "ckpt.pt")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt_out, args.output / "best.pt")
                print(f"  Checkpoint saved → {args.output}/ckpt.pt")
            step += 1
        if step >= args.max_iters:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "metadata": {
                "tokenizer_type": tokenizer.tokenizer_type,
                "tokenizer_hash": tokenizer_hash(tokenizer),
                "vocab_size": tokenizer.vocab_size,
                "git_sha": get_git_sha(),
                "training_step": step,
                "config_name": cfg.get("name", ""),
                "package_versions": get_package_versions(),
            },
        },
        args.output / "final.pt",
    )
    print(f"\n  SFT complete. Best loss: {best_loss:.4f}")
    print(f"  Final model → {args.output}/final.pt")


if __name__ == "__main__":
    main()
