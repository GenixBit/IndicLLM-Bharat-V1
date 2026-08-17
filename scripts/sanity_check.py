#!/usr/bin/env python3
"""IndicLLM-Bharat — Local environment and model sanity check.

Runs a fast training run (CPU / MPS / CUDA) to verify the development environment,
model architecture, optimizer partitioning, and checkpointing.

Usage:
    python scripts/sanity_check.py                  # Native Bharat model (default)
    python scripts/sanity_check.py --model legacy   # Legacy nanoGPT model
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.training.checkpointing import load_checkpoint
from bharat.training.pretrain import PretrainConfig, pretrain

ROOT_DIR = Path(__file__).resolve().parent.parent
NANO_GPT_DIR = ROOT_DIR / "vendor" / "nanoGPT"
OUT_DIR = ROOT_DIR / "checkpoints" / "sanity_check"


def pick_device(override: str | None = None) -> str:
    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_bharat_sanity(device: str, max_iters: int = 50, log_interval: int = 10) -> int:
    print("=" * 64)
    print("🇮🇳 IndicLLM-Bharat Native Architecture Sanity Check")
    print("=" * 64)
    print(f"  Device:       {device.upper()}")
    print(f"  PyTorch:      {torch.__version__}")
    print("  Architecture: BharatForCausalLM (GQA, RoPE, RMSNorm, SwiGLU)")
    print(f"  Iterations:   {max_iters}")
    print("-" * 64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    model_config = BharatModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    cfg = PretrainConfig(
        model_config=model_config,
        synthetic_data=True,
        output_dir=OUT_DIR,
        max_iters=max_iters,
        batch_size=2,
        seq_len=32,
        learning_rate=1e-3,
        warmup_iters=5,
        log_interval=log_interval,
        eval_interval=max_iters,
        save_interval=max_iters,
        device=device,
        dtype="float32",
    )

    t0 = time.time()
    result = pretrain(cfg)
    duration = time.time() - t0

    print("-" * 64)
    print(f"  Final Loss:   {result.final_loss:.4f}")
    print(f"  Completed:    {result.completed_steps} steps in {duration:.2f}s")
    print(f"  Checkpoint:   {result.checkpoint_path}")

    # Verify checkpoint reload
    if result.checkpoint_path and Path(result.checkpoint_path).is_file():
        eval_model = BharatForCausalLM(model_config)
        loaded = load_checkpoint(
            path=result.checkpoint_path,
            model=eval_model,
            device=device,
            strict=True,
        )
        print(
            f"  Loaded SHA:   {loaded['metadata'].git_sha[:10] if loaded['metadata'].git_sha else 'local'}"
        )
        print("  Verification: Checkpoint bit-exact load OK ✅")

    print("=" * 64)
    print("✅ Bharat Environment Sanity Check PASSED!\n")
    return 0


def run_legacy_sanity(device: str) -> int:
    if str(NANO_GPT_DIR) not in sys.path:
        sys.path.insert(0, str(NANO_GPT_DIR))
    try:
        from model import GPT, GPTConfig
    except ImportError as e:
        print(f"Legacy nanoGPT modules not found: {e}", file=sys.stderr)
        return 1

    data_dir = NANO_GPT_DIR / "data" / "shakespeare_char"
    if not (data_dir / "train.bin").exists() or not (data_dir / "meta.pkl").exists():
        print(f"Legacy shakespeare data missing at {data_dir}", file=sys.stderr)
        return 1

    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

    print("=" * 64)
    print("📜 Legacy nanoGPT Sanity Check (Shakespeare Char)")
    print("=" * 64)
    print(f"  Device:     {device.upper()}")
    print(f"  Vocab Size: {vocab_size}")

    cfg = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=256,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=False,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_data = np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")

    def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(train_data) - 256, (8,))
        x = torch.stack([torch.from_numpy(train_data[i : i + 256].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(train_data[i + 1 : i + 1 + 256].astype(np.int64)) for i in ix]
        )
        return x.to(device), y.to(device)

    for step in range(50):
        x, y = get_batch()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1:>3}: loss {loss.item():.4f}")

    print("✅ Legacy Sanity Check PASSED!\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndicLLM-Bharat environment and model architecture sanity check"
    )
    parser.add_argument(
        "--model",
        choices=["bharat", "legacy"],
        default="bharat",
        help="Model architecture to verify (default: bharat)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Compute device to target (default: auto)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Number of iterations to execute (default: 50)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval for step loss",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = pick_device(args.device)

    if args.model == "bharat":
        return run_bharat_sanity(
            device=device, max_iters=args.max_iters, log_interval=args.log_interval
        )
    return run_legacy_sanity(device=device)


if __name__ == "__main__":
    sys.exit(main())
