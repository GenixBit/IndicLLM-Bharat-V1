#!/usr/bin/env python3
"""
nanoGPT-style pretraining with W&B logging.

Usage:
  python train/pretrain.py --config configs/gpt2-124m.yaml
  python train/pretrain.py --config configs/gpt2-124m.yaml --max-iters 100  # smoke test
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Force line-buffered stdout so nohup logs appear immediately
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.utils import ensure_dir, get_device_preference, init_wandb, load_config
from bharat.tokenizer import load_tokenizer
from bharat.tokenizer.metadata import tokenizer_hash
from bharat.training.checkpointing import get_git_sha, get_package_versions


class GPTConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(0.0)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        return self.c_proj(self.dropout(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(0.0),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert self.config.block_size >= T
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_config(cls, model_cfg: dict) -> GPT:
        return cls(GPTConfig(**model_cfg))


def get_lr(it: int, cfg: dict) -> float:
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * it / cfg["warmup_iters"]
    if it > cfg["lr_decay_iters"]:
        return cfg["min_lr"]
    decay_ratio = (it - cfg["warmup_iters"]) / (cfg["lr_decay_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


def load_bin(path: Path) -> torch.Tensor:
    # Cast uint16 → int64: cross_entropy requires int64 targets
    data = __import__("numpy").memmap(path, dtype=__import__("numpy").uint16, mode="r")
    return torch.from_numpy(data.astype(__import__("numpy").int64))


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device, ctx):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        arr = data[split]
        for k in range(eval_iters):
            ix = torch.randint(len(arr) - block_size, (batch_size,))
            x = torch.stack([arr[i : i + block_size] for i in ix])
            y = torch.stack([arr[i + 1 : i + 1 + block_size] for i in ix])
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=None, help="Override max_iters")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    if args.max_iters is not None:
        train_cfg["max_iters"] = args.max_iters

    device = get_device_preference()
    print(f"Device      : {device.upper()}")
    print(f"Config      : {args.config}")
    print(f"Max iters   : {train_cfg['max_iters']}")
    print(
        f"Batch size  : {train_cfg['batch_size']} x grad_accum {train_cfg['gradient_accumulation_steps']}"
    )
    print(f"Block size  : {model_cfg['block_size']}")
    sys.stdout.flush()
    dtype_name = train_cfg.get("dtype", "float32")
    if device == "mps" and dtype_name == "bfloat16":
        dtype_name = "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype_name
    ]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device.split(":")[0], dtype=ptdtype)
    )

    data_cfg = cfg["data"]
    meta_path = Path(data_cfg["meta_pkl"])
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model_cfg["vocab_size"] = meta["vocab_size"]

    train_data = load_bin(Path(data_cfg["train_bin"]))
    val_data = load_bin(Path(data_cfg["val_bin"]))
    data = {"train": train_data, "val": val_data}

    out_dir = ensure_dir(cfg["checkpoint"]["out_dir"])

    # Tokenizer
    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = load_tokenizer(tok_src)
    print(f"  Tokenizer: {tokenizer.tokenizer_type} (vocab={tokenizer.vocab_size})")

    # Validate on resume
    init_from = cfg.get("checkpoint", {}).get("init_from", "scratch")
    if init_from == "resume":
        ckpt_path = out_dir / "ckpt.pt"
        if ckpt_path.exists():
            old_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            old_meta = old_ckpt.get("metadata")
            if old_meta and old_meta.get("tokenizer_hash"):
                old_hash = old_meta["tokenizer_hash"]
                cur_hash = tokenizer_hash(tokenizer)
                if old_hash != cur_hash:
                    raise ValueError(
                        f"Tokenizer mismatch on resume: checkpoint has hash="
                        f"{old_hash[:12]}..., current={cur_hash[:12]}...\n"
                        f"Use the same tokenizer that was used during training."
                    )
            elif old_meta:
                print("  Warning: checkpoint has metadata but no tokenizer_hash — skipping validation.")
            else:
                print("  Warning: checkpoint has no metadata section — skipping tokenizer validation.")

    init_wandb(cfg)
    wandb = None
    if os.environ.get("WANDB_API_KEY") and cfg.get("wandb", {}).get("enabled"):
        import wandb as _wandb

        wandb = _wandb

    model = GPT.from_config(model_cfg).to(device)
    if train_cfg.get("compile") and device == "cuda":
        print("  torch.compile enabled (CUDA)")
        model = torch.compile(model)

    optimizer = model.configure_optimizers if hasattr(model, "configure_optimizers") else None
    if optimizer is None:
        import inspect

        # Use raw uncompiled model for named_modules (torch.compile wraps with _orig_mod)
        raw_model = getattr(model, "_orig_mod", model)

        decay = set()
        no_decay = set()
        for mn, m in raw_model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if (
                    pn.endswith("bias")
                    or pn.endswith("weight")
                    and isinstance(m, (nn.LayerNorm, nn.Embedding))
                ):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        param_dict = {pn: p for pn, p in raw_model.named_parameters()}
        decay = decay & param_dict.keys()
        no_decay = no_decay & param_dict.keys()
        optim_groups = [
            {
                "params": [param_dict[p] for p in sorted(decay)],
                "weight_decay": train_cfg["weight_decay"],
            },
            {"params": [param_dict[p] for p in sorted(no_decay)], "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_cfg["learning_rate"], betas=(0.9, 0.95), fused=fused
        )

    block_size = model_cfg["block_size"]
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]

    t0 = time.time()
    for it in range(train_cfg["max_iters"] + 1):
        lr = get_lr(it, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % train_cfg["eval_interval"] == 0 and it > 0:
            losses = estimate_loss(
                model, data, block_size, batch_size, train_cfg["eval_iters"], device, ctx
            )
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb:
                wandb.log(
                    {"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr}, step=it
                )
            if it > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": it,
                    "config": cfg,
                    "metadata": {
                        "tokenizer_type": tokenizer.tokenizer_type,
                        "tokenizer_hash": tokenizer_hash(tokenizer),
                        "vocab_size": tokenizer.vocab_size,
                        "git_sha": get_git_sha(),
                        "training_step": it,
                        "config_name": cfg.get("name", ""),
                        "package_versions": get_package_versions(),
                    },
                }
                torch.save(ckpt, out_dir / "ckpt.pt")

        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            ix = torch.randint(len(train_data) - block_size, (batch_size,))
            x = torch.stack([train_data[i : i + block_size] for i in ix])
            y = torch.stack([train_data[i + 1 : i + 1 + block_size] for i in ix])
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()

        if it % train_cfg["log_interval"] == 0:
            dt = time.time() - t0
            print(f"iter {it}: loss {loss.item() * grad_accum:.4f}, time {dt * 1000:.0f}ms")
            t0 = time.time()

    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "metadata": {
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_hash": tokenizer_hash(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "git_sha": get_git_sha(),
            "training_step": train_cfg["max_iters"],
            "config_name": cfg.get("name", ""),
            "package_versions": get_package_versions(),
        },
    }, out_dir / "final.pt")
    print(f"Training complete. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
