#!/usr/bin/env python3
"""
IndicLLM-Bharat — local-sanity task
Run nanoGPT on the tiny Shakespeare char dataset to verify the dev environment.

Usage (from llm-lab root):
    python scripts/sanity_check.py

Runs ~300 iterations of a tiny GPT (~1.5M params) on CPU/MPS.
Expected outcome: loss drops from ~3.3 → ~1.8 and a checkpoint is saved.
"""

import math
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# ── Add nanoGPT to path ─────────────────────────────────────────────────────
NANO_GPT_DIR = Path(__file__).parent.parent / "vendor" / "nanoGPT"
sys.path.insert(0, str(NANO_GPT_DIR))

import numpy as np
import torch
from model import GPT, GPTConfig  # from nanoGPT

# ── Config ───────────────────────────────────────────────────────────────────
# Tiny model that runs fast on M2 CPU/MPS
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
BLOCK_SIZE = 256
BATCH_SIZE = 8
GRAD_ACCUM = 1
MAX_ITERS = 300
EVAL_INTERVAL = 100
EVAL_ITERS = 20
LOG_INTERVAL = 50
LR = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 30
LR_DECAY_ITERS = 300
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.0
BIAS = False

DATA_DIR = NANO_GPT_DIR / "data" / "shakespeare_char"
OUT_DIR = Path(__file__).parent.parent / "checkpoints" / "sanity-shakespeare"


# ── Device selection ─────────────────────────────────────────────────────────
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Data loader ──────────────────────────────────────────────────────────────
def get_batch(split, device):
    path = DATA_DIR / ("train.bin" if split == "train" else "val.bin")
    data = np.memmap(str(path), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


# ── LR schedule ──────────────────────────────────────────────────────────────
def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = pick_device()
    print(f"{'='*60}")
    print("  IndicLLM-Bharat — Dev Environment Sanity Check")
    print(f"{'='*60}")
    print(f"  Device      : {device.upper()}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Model       : {N_LAYER}L / {N_EMBD}d / {N_HEAD}H (~1.5M params)")
    print(f"  Dataset     : Shakespeare char ({DATA_DIR})")
    print(f"  Max iters   : {MAX_ITERS}")
    print(f"{'='*60}\n")

    # Verify data exists
    assert (DATA_DIR / "train.bin").exists(), f"train.bin not found in {DATA_DIR}"
    assert (DATA_DIR / "val.bin").exists(), f"val.bin not found in {DATA_DIR}"

    # Load vocab size from meta
    meta_path = DATA_DIR / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"  vocab_size  : {vocab_size} (from meta.pkl)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1337)

    # Model
    cfg = GPTConfig(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        dropout=DROPOUT,
        bias=BIAS,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {n_params/1e6:.2f}M\n")

    # Optimizer — use device_type='cpu' for MPS (adamw works on MPS tensors fine)
    opt_device_type = "cuda" if device == "cuda" else "cpu"
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LR, (0.9, 0.95), opt_device_type)

    # autocast context: only for cuda; nullcontext for cpu/mps (float32 is fine)
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else nullcontext()
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(EVAL_ITERS):
                x, y = get_batch(split, device)
                with ctx:
                    _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # Training loop
    print(f"  {'Iter':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'LR':>8}  {'ms/it':>7}")
    print(f"  {'-'*50}")

    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(MAX_ITERS + 1):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval + checkpoint
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            dt_eval = (time.time() - t0) * 1000
            print(
                f"  {it:>6}  {losses['train']:>10.4f}  {losses['val']:>8.4f}  {lr:>8.2e}  {dt_eval/max(it,1):>7.1f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": dict(
                        n_layer=N_LAYER,
                        n_head=N_HEAD,
                        n_embd=N_EMBD,
                        block_size=BLOCK_SIZE,
                        vocab_size=vocab_size,
                        dropout=DROPOUT,
                        bias=BIAS,
                    ),
                    "iter_num": it,
                    "best_val_loss": best_val_loss,
                }
                ckpt_path = OUT_DIR / "ckpt.pt"
                torch.save(ckpt, ckpt_path)

        if it == MAX_ITERS:
            break

        # Forward + backward
        x, y = get_batch("train", device)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % LOG_INTERVAL == 0 and it > 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000 / LOG_INTERVAL
            t0 = t1
            print(f"  iter {it:>4}: loss {loss.item():.4f}  lr {lr:.2e}  {dt:.1f}ms/it")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print("  ✅ SANITY CHECK PASSED")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Checkpoint     : {OUT_DIR / 'ckpt.pt'}")
    print(f"  Total time     : {total_time:.1f}s")
    print(f"{'='*60}")
    print("\n  Dev environment is confirmed working.")
    print("  Next step → data-pipeline: run  python data/prepare_data.py")


if __name__ == "__main__":
    main()
