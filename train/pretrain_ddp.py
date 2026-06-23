#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — DDP (DistributedDataParallel) Pretraining

Multi-GPU version of pretrain.py using PyTorch DDP.
Launch with torchrun:
  torchrun --standalone --nproc_per_node=4 train/pretrain_ddp.py --config configs/gpt2-350m.yaml

Or use the wrapper:
  bash infra/ddp_launch.sh --gpus 4 --config configs/gpt2-350m.yaml
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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT, GPTConfig, get_lr
from train.utils import init_wandb, load_config

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def load_bin(path: Path) -> np.ndarray:
    return np.memmap(str(path), dtype=np.uint16, mode="r")


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters, device, ctx, rank):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        arr = data[split]
        for k in range(eval_iters):
            ix = torch.randint(len(arr) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy(arr[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(arr[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    raw.train()
    return out


def main():
    rank, world_size, local_rank = setup_ddp()
    is_master = (rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=Path, required=True)
    parser.add_argument("--max-iters", type=int,  default=None)
    args = parser.parse_args()

    cfg       = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    if args.max_iters:
        train_cfg["max_iters"] = args.max_iters

    device   = f"cuda:{local_rank}"
    ptdtype  = torch.bfloat16
    ctx      = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    if is_master:
        print(f"DDP training: {world_size} GPUs")
        print(f"Config      : {args.config}")
        print(f"Max iters   : {train_cfg['max_iters']}")
        print(f"Eff. batch  : {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps'] * world_size}")

    # Data
    data_cfg = cfg["data"]
    train_arr = load_bin(Path(data_cfg["train_bin"]))
    val_arr   = load_bin(Path(data_cfg["val_bin"]))
    data = {"train": train_arr, "val": val_arr}

    # Meta
    meta_path = Path(data_cfg["meta_pkl"])
    if meta_path.exists():
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model_cfg["vocab_size"] = meta["vocab_size"]

    # Model
    model = GPT(GPTConfig(**model_cfg)).to(device)
    if train_cfg.get("compile"):
        model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    # Optimizer (on raw model)
    raw_model = model.module
    decay, no_decay = set(), set()
    for mn, m in raw_model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)
            else:
                decay.add(fpn)
    param_dict = {pn: p for pn, p in raw_model.named_parameters()}
    decay    = decay    & param_dict.keys()
    no_decay = no_decay & param_dict.keys()
    optim_groups = [
        {"params": [param_dict[p] for p in sorted(decay)],    "weight_decay": train_cfg["weight_decay"]},
        {"params": [param_dict[p] for p in sorted(no_decay)], "weight_decay": 0.0},
    ]
    import inspect
    fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    optimizer = torch.optim.AdamW(
        optim_groups, lr=train_cfg["learning_rate"], betas=(0.9, 0.95), fused=fused
    )

    # W&B (master only)
    wandb = None
    if is_master and os.environ.get("WANDB_API_KEY") and cfg.get("wandb", {}).get("enabled"):
        import wandb as _wandb
        _wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"].get("run_name", "ddp-run"), config=cfg)
        wandb = _wandb

    out_dir = Path(cfg["checkpoint"]["out_dir"])
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    block_size = model_cfg["block_size"]
    batch_size = train_cfg["batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    scaler     = torch.cuda.amp.GradScaler()

    t0 = time.time()
    for it in range(train_cfg["max_iters"] + 1):
        lr = get_lr(it, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval (master only, every eval_interval, skip step 0)
        if is_master and it % train_cfg["eval_interval"] == 0 and it > 0:
            losses = estimate_loss(model, data, block_size, batch_size,
                                   train_cfg["eval_iters"], device, ctx, rank)
            print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iter_num": it, "config": cfg}
            torch.save(ckpt, out_dir / "ckpt.pt")
            if wandb:
                wandb.log({"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr}, step=it)

        optimizer.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            # Sync gradients only on last micro-step
            model.require_backward_grad_sync = (micro == grad_accum - 1)
            ix = torch.randint(len(train_arr) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy(train_arr[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(train_arr[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if is_master and it % train_cfg["log_interval"] == 0:
            dt = time.time() - t0
            tok_per_sec = batch_size * block_size * world_size / dt
            print(f"iter {it}: loss {loss.item()*grad_accum:.4f}  {tok_per_sec:.0f} tok/s  {dt*1000:.0f}ms")
            t0 = time.time()

    if is_master:
        torch.save({"model": raw_model.state_dict(), "config": cfg}, out_dir / "final.pt")
        print(f"Done. Final model → {out_dir}/final.pt")
        if wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
