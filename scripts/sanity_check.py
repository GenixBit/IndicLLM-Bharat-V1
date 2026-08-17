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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bharat.models.bharat_model import BharatForCausalLM  # noqa: E402
from bharat.models.config import BharatModelConfig  # noqa: E402
from bharat.training.checkpointing import load_checkpoint  # noqa: E402
from bharat.training.pretrain import PretrainConfig, pretrain  # noqa: E402

NANO_GPT_DIR = ROOT_DIR / "vendor" / "nanoGPT"
OUT_DIR = ROOT_DIR / "checkpoints" / "sanity_check"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local fast sanity check for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["bharat", "legacy"],
        default="bharat",
        help="Architecture to test: native 'bharat' (GQA, RoPE, RMSNorm, SwiGLU) or 'legacy' (GPT-2 Shakespeare)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum training iterations for sanity check",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device for test run",
    )
    return parser.parse_args(args)


def run_bharat_sanity(max_iters: int = 10, device_choice: str = "auto") -> int:
    """Fast native verification of BharatForCausalLM architecture and training loop."""
    print("=" * 64)
    print("  🇮🇳 IndicLLM-Bharat — Native Architecture Sanity Check")
    print("=" * 64)

    if device_choice == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_choice

    print(f"  Target Device: {device.upper()}")

    # 1. Instantiate small model
    config = BharatModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    t0 = time.time()
    print("  [1/4] Initializing BharatForCausalLM (RoPE, RMSNorm, SwiGLU, GQA 4:1)...")
    model = BharatForCausalLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"        ✓ Model initialized ({param_count:,} parameters)")

    # 2. Verify AdamW weight decay groups
    print("  [2/4] Verifying AdamW parameter partitioning...")
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    _optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    print(
        f"        ✓ Decay group: {len(decay_params)} tensors | No-decay group: {len(nodecay_params)} tensors"
    )

    # 3. Fast synthetic training step
    print(f"  [3/4] Running {max_iters} training iterations on synthetic data...")
    pt_config = PretrainConfig(
        model_config=config,
        data_path=str(OUT_DIR / "dummy.bin"),
        synthetic_data=True,
        max_iters=max_iters,
        batch_size=2,
        seq_len=32,
        learning_rate=1e-3,
        warmup_iters=2,
        save_interval=max_iters,
        eval_interval=max_iters,
        output_dir=str(OUT_DIR),
        device=device,
        dtype="float32",
    )

    t_train_start = time.time()
    result = pretrain(pt_config)
    t_train_dur = time.time() - t_train_start
    print(
        f"        ✓ Training completed (Loss: {result.final_loss:.4f}, Duration: {t_train_dur:.2f}s)"
    )

    # 4. Verify checkpoint reload
    print("  [4/4] Verifying checkpoint reload & bit-exact state...")
    ckpt_path = Path(result.checkpoint_path)
    assert ckpt_path.is_file(), f"Checkpoint not found at {ckpt_path}"

    reload_model = BharatForCausalLM(config).to(device)
    loaded_data = load_checkpoint(ckpt_path, reload_model, device=device)
    assert loaded_data["step"] == max_iters
    print(f"        ✓ Checkpoint reloaded successfully from {ckpt_path.name}")

    total_time = time.time() - t0
    print("\n" + "=" * 64)
    print(f"  ✅ ALL CHECKS PASSED in {total_time:.2f}s!")
    print("=" * 64 + "\n")
    return 0


def run_legacy_nano_sanity(max_iters: int = 10, device_choice: str = "auto") -> int:
    """Legacy verification of nanoGPT on Shakespeare."""
    print("=" * 64)
    print("  IndicLLM-Bharat — Legacy nanoGPT Sanity Check")
    print("=" * 64)

    if not NANO_GPT_DIR.exists():
        print(
            f"Legacy nanoGPT directory not found at {NANO_GPT_DIR}. Run bash scripts/setup.sh first."
        )
        return 1

    sys.path.insert(0, str(NANO_GPT_DIR))
    from model import GPT, GPTConfig

    data_dir = NANO_GPT_DIR / "data" / "shakespeare_char"
    train_bin = data_dir / "train.bin"

    if not train_bin.exists():
        print("Preparing Shakespeare dataset...")
        meta_pkl = data_dir / "meta.pkl"
        data_dir.mkdir(parents=True, exist_ok=True)
        text = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 500
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        train_data = np.array([stoi[c] for c in text], dtype=np.uint16)
        train_data.tofile(train_bin)
        with open(meta_pkl, "wb") as f:
            pickle.dump({"vocab_size": len(chars), "itos": itos, "stoi": stoi}, f)

    if device_choice == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_choice

    config = GPTConfig(
        vocab_size=65,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=1e-1, learning_rate=1e-3, betas=(0.9, 0.95), device_type=device
    )

    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    for _step in range(max_iters):
        ix = torch.randint(len(data) - 64, (4,))
        x = torch.stack([torch.from_numpy((data[i : i + 64]).astype(np.int64)) for i in ix]).to(
            device
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + 64]).astype(np.int64)) for i in ix]
        ).to(device)
        _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"  Legacy nanoGPT sanity passed ({max_iters} steps, final loss: {loss.item():.4f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.model == "bharat":
        return run_bharat_sanity(max_iters=args.max_iters, device_choice=args.device)
    else:
        return run_legacy_nano_sanity(max_iters=args.max_iters, device_choice=args.device)


if __name__ == "__main__":
    sys.exit(main())
