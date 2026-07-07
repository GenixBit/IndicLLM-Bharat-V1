#!/usr/bin/env python3
# ruff: noqa: E402, N806
"""
nanoGPT-style pretraining with precise step semantics.

Step semantics
--------------
- completed_steps : number of optimizer steps already finished.
- next_step       : the optimizer-step index to execute next.
                    Always equal to completed_steps during a fresh run.
- Checkpoints are saved AFTER each completed optimizer step.
- Resuming starts at checkpoint[next_step] and never repeats a step.
- LR scheduling uses the global step index (0-based).

Legacy compatibility
--------------------
Pre-Milestone-1.3 checkpoints used the key ``iter_num`` to store the
last-evaluated iteration.  During resume those are converted to
``next_step`` automatically when ``compatibility_mode: legacy`` is
set in the checkpoint config.

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

from bharat.tokenizer import load_tokenizer
from bharat.tokenizer.metadata import tokenizer_hash
from bharat.training.checkpointing import get_git_sha, get_package_versions
from train.utils import ensure_dir, get_device_preference, init_wandb, load_config


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
        _B, T = idx.size()
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
    # Cast uint16 -> int64: cross_entropy requires int64 targets
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


def _build_optimizer(model, train_cfg):
    import inspect

    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, _p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or (
                pn.endswith("weight") and isinstance(m, nn.LayerNorm | nn.Embedding)
            ):
                no_decay.add(fpn)
            else:
                decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay_set = decay & param_dict.keys()
    no_decay_set = no_decay & param_dict.keys()
    optim_groups = [
        {
            "params": [param_dict[p] for p in sorted(decay_set)],
            "weight_decay": train_cfg["weight_decay"],
        },
        {"params": [param_dict[p] for p in sorted(no_decay_set)], "weight_decay": 0.0},
    ]
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        optim_groups, lr=train_cfg["learning_rate"], betas=(0.9, 0.95), fused=fused
    )
    return optimizer


def _build_checkpoint(model, optimizer, completed_steps, tokenizer, cfg, _grad_accum):
    import random

    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "next_step": completed_steps,
        "config": cfg,
        "final_loss": None,
        "metadata": {
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_hash": tokenizer_hash(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "git_sha": get_git_sha(),
            "training_step": completed_steps,
            "config_name": cfg.get("name", ""),
            "package_versions": get_package_versions(),
        },
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state().tolist(),
            "cuda": {
                str(i): torch.cuda.get_rng_state(i).tolist()
                for i in range(torch.cuda.device_count())
            }
            if torch.cuda.is_available()
            else {},
        },
    }


def _validate_pretrain_config(cfg: dict) -> None:
    errors: list[str] = []
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    max_iters = train_cfg.get("max_iters", 0)
    if max_iters <= 0:
        errors.append(f"training.max_iters must be > 0, got {max_iters}")

    batch_size = train_cfg.get("batch_size", 0)
    if batch_size <= 0:
        errors.append(f"training.batch_size must be > 0, got {batch_size}")

    block_size = model_cfg.get("block_size", 0)
    if block_size <= 0:
        errors.append(f"model.block_size must be > 0, got {block_size}")

    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    if grad_accum <= 0:
        errors.append(f"training.gradient_accumulation_steps must be > 0, got {grad_accum}")

    eval_interval = train_cfg.get("eval_interval", 100)
    if eval_interval <= 0:
        errors.append(f"training.eval_interval must be > 0, got {eval_interval}")

    save_interval = train_cfg.get("save_interval", eval_interval)
    if save_interval <= 0:
        errors.append(f"training.save_interval must be > 0, got {save_interval}")

    log_interval = train_cfg.get("log_interval", 10)
    if log_interval <= 0:
        errors.append(f"training.log_interval must be > 0, got {log_interval}")

    warmup_iters = train_cfg.get("warmup_iters", 0)
    if warmup_iters < 0:
        errors.append(f"training.warmup_iters must be >= 0, got {warmup_iters}")

    lr_decay_iters = train_cfg.get("lr_decay_iters", max_iters)
    if lr_decay_iters <= warmup_iters:
        errors.append(
            f"training.lr_decay_iters ({lr_decay_iters}) must be > "
            f"training.warmup_iters ({warmup_iters})"
        )

    if errors:
        raise ValueError("Pretrain config validation failed:\n" + "\n".join(errors))


def train_from_config(
    cfg: dict,
    max_iters: int | None = None,
) -> dict[str, int | float | str | None]:
    """Run the pretraining loop.

    Args:
        cfg: Training configuration dict.
        max_iters: Optional override for maximum training iterations.

    Returns:
        dict with keys: final_loss, completed_steps, next_step, output_dir.
    """
    _validate_pretrain_config(cfg)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"].copy()
    if max_iters is not None:
        train_cfg["max_iters"] = max_iters
    else:
        max_iters = train_cfg.get("max_iters", 0)

    if max_iters <= 0:
        raise ValueError(
            f"max_iters must be > 0 for training, got {max_iters}. "
            "Use a positive value to run training steps."
        )

    device = get_device_preference()
    print(f"Device      : {device.upper()}")
    print(f"Max iters   : {max_iters}")
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

    block_size = model_cfg["block_size"]
    if len(train_data) <= block_size:
        raise ValueError(
            f"Training data length ({len(train_data)}) must be > block_size ({block_size})"
        )
    if len(val_data) <= block_size:
        raise ValueError(
            f"Validation data length ({len(val_data)}) must be > block_size ({block_size})"
        )
    data = {"train": train_data, "val": val_data}

    out_dir = ensure_dir(cfg["checkpoint"]["out_dir"])

    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = load_tokenizer(tok_src)
    print(f"  Tokenizer: {tokenizer.tokenizer_type} (vocab={tokenizer.vocab_size})")

    init_wandb(cfg)
    wandb = None
    if os.environ.get("WANDB_API_KEY") and cfg.get("wandb", {}).get("enabled"):
        import wandb as _wandb

        wandb = _wandb

    model = GPT.from_config(model_cfg).to(device)

    init_from = cfg.get("checkpoint", {}).get("init_from", "scratch")
    compatibility_mode = cfg.get("checkpoint", {}).get("compatibility_mode")
    next_step = 0
    rng_state = None

    optimizer = _build_optimizer(model, train_cfg)

    # Handle resume
    if init_from == "resume":
        ckpt_path = out_dir / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found at {ckpt_path}. "
                "Use init_from: scratch to start fresh."
            )

        print(f"  Resuming from {ckpt_path}")
        old_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # --- Validate metadata ---
        old_meta = old_ckpt.get("metadata")
        if old_meta and old_meta.get("tokenizer_hash"):
            cur_hash = tokenizer_hash(tokenizer)
            if old_meta["tokenizer_hash"] != cur_hash:
                raise ValueError(
                    f"Tokenizer mismatch on resume: checkpoint hash="
                    f"{old_meta['tokenizer_hash'][:12]}..., "
                    f"current={cur_hash[:12]}...\n"
                    f"Use the same tokenizer that was used during training."
                )
        elif old_meta:
            raise ValueError(
                "Checkpoint has metadata but no tokenizer_hash. "
                "Cannot safely validate tokenizer compatibility."
            )
        else:
            raise ValueError(
                "Checkpoint has no metadata section. "
                "Cannot safely validate tokenizer compatibility."
            )

        # --- Resolve step ---
        if "next_step" in old_ckpt:
            next_step = old_ckpt["next_step"]
        elif compatibility_mode == "legacy":
            # Legacy checkpoint: was saved with iter_num = the step just evaluated
            # (NOT the step just completed).  We stored iter_num == it where
            # it was the loop variable BEFORE the optimizer step.
            # Legacy resume started at iter_num, which repeated that step.
            # Clean resume must start at iter_num + 1 to skip the repeated work.
            legacy_iter = old_ckpt.get("iter_num", 0)
            next_step = legacy_iter + 1
            print(
                f"  Legacy checkpoint: converting iter_num={legacy_iter} -> next_step={next_step}"
            )
        else:
            raise ValueError(
                "Checkpoint missing 'next_step' key and compatibility_mode is not 'legacy'. "
                "Set checkpoint.compatibility_mode: legacy to load pre-Milestone-1.3 checkpoints."
            )

        print(f"  Resuming at step {next_step}")

        # Reject checkpoints ahead of the requested target
        if next_step > max_iters:
            raise ValueError(
                f"Checkpoint next_step={next_step} exceeds requested max_iters={max_iters}"
            )

        # Already at target → no-op result (skip model/optimizer/RNG loading)
        if next_step == max_iters:
            print(
                f"  Already at next_step={next_step} == max_iters={max_iters}, no training needed"
            )
            return {
                "final_loss": None,
                "completed_steps": next_step,
                "next_step": next_step,
                "output_dir": str(out_dir),
            }

        # --- Load model state (strict) ---
        model_state = old_ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in model_state):
            clean_state = {}
            for k, v in model_state.items():
                clean_state[k.replace("_orig_mod.", "")] = v
            model_state = clean_state

        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=True)
        assert not missing_keys, f"Missing model keys on resume: {missing_keys}"
        assert not unexpected_keys, f"Unexpected model keys on resume: {unexpected_keys}"

        # --- Load optimizer state ---
        if "optimizer" not in old_ckpt:
            raise ValueError("Resume checkpoint missing optimizer state. Cannot resume.")
        optimizer.load_state_dict(old_ckpt["optimizer"])

        # --- Restore RNG state ---
        if "rng_state" not in old_ckpt:
            raise ValueError("Resume checkpoint missing rng_state. Cannot resume.")
        rng_state = old_ckpt["rng_state"]
        import random

        random.setstate(rng_state.get("python", random.getstate()))
        if rng_state.get("torch"):
            torch.set_rng_state(torch.tensor(rng_state["torch"], dtype=torch.uint8))
        if torch.cuda.is_available():
            for dev_id, dev_state in rng_state.get("cuda", {}).items():
                torch.cuda.set_rng_state(
                    torch.tensor(dev_state, dtype=torch.uint8),
                    device=int(dev_id),
                )
    else:
        print("  Initializing from scratch")

    if train_cfg.get("compile") and device == "cuda":
        print("  torch.compile enabled (CUDA)")
        model = torch.compile(model)

    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    eval_interval = train_cfg.get("eval_interval", 100)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", eval_interval)

    t0 = time.time()
    completed_steps = next_step  # steps already done before this run

    # Reject checkpoints ahead of the requested target
    if next_step > max_iters:
        raise ValueError(
            f"Checkpoint next_step={next_step} exceeds requested max_iters={max_iters}"
        )

    # Already at target → no-op result
    if next_step == max_iters:
        print(f"  Already at next_step={next_step} == max_iters={max_iters}, no training needed")
        return {
            "final_loss": None,
            "completed_steps": completed_steps,
            "next_step": next_step,
            "output_dir": str(out_dir),
        }

    for step in range(next_step, max_iters):
        # Set learning rate for this step
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward / backward / optimizer
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

        # Step is complete
        completed_steps = step + 1

        # Logging
        if step % log_interval == 0 or step == next_step:
            dt = time.time() - t0
            print(f"step {step}: loss {loss.item() * grad_accum:.4f}, time {dt * 1000:.0f}ms")
            t0 = time.time()

        # Evaluation
        if completed_steps % eval_interval == 0:
            losses = estimate_loss(
                model, data, block_size, batch_size, train_cfg["eval_iters"], device, ctx
            )
            print(
                f"step {step} (completed {completed_steps}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb:
                wandb.log(
                    {"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr},
                    step=completed_steps,
                )

        # Checkpoint after step
        if completed_steps % save_interval == 0:
            ckpt = _build_checkpoint(model, optimizer, completed_steps, tokenizer, cfg, grad_accum)
            torch.save(ckpt, out_dir / "ckpt.pt")

    # Final checkpoint
    final_loss_val = loss.item() * grad_accum
    final_ckpt = _build_checkpoint(model, optimizer, completed_steps, tokenizer, cfg, grad_accum)
    final_ckpt["final_loss"] = final_loss_val
    torch.save(final_ckpt, out_dir / "final.pt")
    print(f"Training complete. Checkpoints in {out_dir}")

    return {
        "final_loss": final_loss_val,
        "completed_steps": completed_steps,
        "next_step": completed_steps,
        "output_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=None, help="Override max_iters")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_from_config(cfg, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
