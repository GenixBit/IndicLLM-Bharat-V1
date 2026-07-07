#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — DPO (Direct Preference Optimization)

Applies DPO alignment on top of an SFT checkpoint
with per-sample response masking.

Usage:
  python train/dpo.py \
    --sft-checkpoint checkpoints/gpt2-124m-sft/final.pt \
    --data data/dpo/preferences.jsonl \
    --output checkpoints/gpt2-124m-dpo

Data format (JSONL):
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.tokenizer import load_tokenizer
from bharat.tokenizer.metadata import tokenizer_hash
from bharat.training.checkpointing import get_git_sha, get_package_versions
from train.pretrain import GPT, GPTConfig
from train.utils import get_device_preference, init_wandb, load_config


class DPODataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int, pad_token_id: int = 0):
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.pairs: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                self.pairs.append(
                    {
                        "prompt": tokenizer.encode(item["prompt"], add_special_tokens=False),
                        "chosen": tokenizer.encode(item["chosen"], add_special_tokens=False),
                        "rejected": tokenizer.encode(item["rejected"], add_special_tokens=False),
                    }
                )
        print(f"  Loaded {len(self.pairs)} preference pairs from {jsonl_path}")

    def _pad(self, ids: list[int]) -> torch.Tensor:
        ids = ids[: self.block_size]
        ids += [self.pad_token_id] * (self.block_size - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        prompt = p["prompt"]
        prompt_len = len(prompt)
        chosen_ids = self._pad(prompt + p["chosen"])
        rejected_ids = self._pad(prompt + p["rejected"])
        return chosen_ids, rejected_ids, prompt_len


def log_probs(model, ids: torch.Tensor, prompt_lens: torch.Tensor, ctx) -> torch.Tensor:
    """Compute sum of log-probs over the response tokens only, with per-sample masking."""
    with ctx:
        logits, _ = model(ids)
    log_p = F.log_softmax(logits, dim=-1)
    # Gather log-probs of actual next tokens
    tokens = ids[:, 1:].unsqueeze(-1)
    lp = log_p[:, :-1].gather(-1, tokens).squeeze(-1)
    # Per-sample mask: 0 for prompt tokens, 1 for response tokens
    B, T = lp.shape
    arange = torch.arange(T, device=lp.device).unsqueeze(0).expand(B, -1)
    mask = (arange >= prompt_lens.unsqueeze(-1)).float()
    return (lp * mask).sum(-1)


def dpo_loss(
    policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta: float = 0.1
) -> torch.Tensor:
    """Standard DPO loss."""
    chosen_ratio = policy_chosen_lp - ref_chosen_lp
    rejected_ratio = policy_rejected_lp - ref_rejected_lp
    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()


def main():
    parser = argparse.ArgumentParser(description="IndicLLM DPO alignment")
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/gpt2-124m.yaml"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/gpt2-124m-dpo"))
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature — higher = stronger preference signal",
    )
    args = parser.parse_args()

    device = get_device_preference()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    print(f"\n{'=' * 60}")
    print("  IndicLLM DPO Alignment")
    print(f"  SFT  : {args.sft_checkpoint}")
    print(f"  Beta : {args.beta}")
    print(f"  Device: {device.upper()}")
    print(f"{'=' * 60}\n")

    # Policy model (trainable)
    ckpt = torch.load(args.sft_checkpoint, map_location=device, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if "model" in saved_cfg:
        model_cfg = saved_cfg["model"]
    policy = GPT(GPTConfig(**model_cfg)).to(device)
    policy.load_state_dict(ckpt["model"])

    # Reference model (frozen copy of SFT)
    ref = GPT(GPTConfig(**model_cfg)).to(device)
    ref.load_state_dict(ckpt["model"])
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = load_tokenizer(tok_src)
    print(f"  Tokenizer: {tokenizer.tokenizer_type} (vocab={tokenizer.vocab_size})")

    block_size = model_cfg.get("block_size", 1024)
    dataset = DPODataset(args.data, tokenizer, block_size, pad_token_id=tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    ctx = (
        nullcontext()
        if device in ("cpu", "mps")
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    init_wandb({**cfg, "wandb": {**cfg.get("wandb", {}), "run_name": "dpo"}})

    policy.train()
    step = 0
    for _epoch in range(100):
        for chosen, rejected, prompt_len in loader:
            if step >= args.max_iters:
                break
            chosen = chosen.to(device)
            rejected = rejected.to(device)
            pl = prompt_len.to(device)

            policy_chosen_lp = log_probs(policy, chosen, pl, ctx)
            policy_rejected_lp = log_probs(policy, rejected, pl, ctx)
            with torch.no_grad():
                ref_chosen_lp = log_probs(ref, chosen, pl, ctx)
                ref_rejected_lp = log_probs(ref, rejected, pl, ctx)

            loss = dpo_loss(
                policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, args.beta
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 50 == 0:
                reward_acc = (policy_chosen_lp > policy_rejected_lp).float().mean().item()
                print(f"  step {step:>4}: loss {loss.item():.4f}  reward_acc {reward_acc:.2%}")

            if step % 500 == 0 and step > 0:
                torch.save(
                    {
                        "model": policy.state_dict(),
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
                    },
                    args.output / "ckpt.pt",
                )
            step += 1
        if step >= args.max_iters:
            break

    torch.save(
        {
            "model": policy.state_dict(),
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
    print(f"\n  DPO complete → {args.output}/final.pt")


if __name__ == "__main__":
    main()
