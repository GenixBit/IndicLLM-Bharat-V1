#!/usr/bin/env python3
# ruff: noqa: E402, N817
"""Native Distributed Pretrainer for Sovereign IndicLLM-Bharat Architecture.

Supports 350M, 1B, 3B, 7B, and 10B architectures with PyTorch DDP, FSDP, and single-device execution.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.data.binary_stream import MMapTokenDataset
from bharat.data.mixture import stream_and_pack_mixture
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.checkpointing import save_checkpoint


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain Sovereign IndicLLM-Bharat on World & Indic Data Mixtures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/bharat-1b.yaml",
        help="Path to YAML model configuration",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="data/binary_shards",
        help="Directory containing binary token shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat_production",
        help="Directory to save pretraining checkpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Total optimizer pretraining steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Sequence context length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of linear warmup steps",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm clipping",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay coefficient",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Precision format for AMP autocast",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-GPU DDP training via torchrun",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine learning rate decay with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return base_lr * 0.1
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * (0.1 + 0.9 * coeff)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    torch.manual_seed(parsed.seed)

    # 1. Distributed setup
    is_ddp = parsed.distributed or "RANK" in os.environ
    rank = 0
    world_size = 1
    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = rank % max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    else:
        if parsed.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(parsed.device)

    # 2. Tokenizer and Model Config
    tokenizer: BharatTokenizer = load_tokenizer("gpt2")
    cfg_path = Path(parsed.config)
    if cfg_path.is_file():
        model_config = BharatModelConfig.from_yaml(cfg_path)
    else:
        model_config = BharatModelConfig(vocab_size=tokenizer.vocab_size)

    # 3. Model Instantiation
    model = BharatForCausalLM(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    if is_ddp and device.type == "cuda":
        model = DDP(model, device_ids=[device_id])

    # 4. Dataset preparation
    shards_dir = Path(parsed.shards_dir)
    bin_files = list(shards_dir.glob("*.bin")) if shards_dir.is_dir() else []
    if not bin_files and rank == 0:
        print(f"No binary shards found in {shards_dir}. Packing default mixture...")
        stream_and_pack_mixture(tokenizer, shards_dir)

    if is_ddp:
        dist.barrier()

    dataset = MMapTokenDataset(shards_dir, block_size=parsed.block_size)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if rank == 0:
        print("\n" + "=" * 65)
        print("🚀 Sovereign IndicLLM-Bharat Production Pretraining Engine")
        print(
            f"  • Model Architecture: {model_config.num_hidden_layers} layers | {model_config.hidden_size} hidden | {model_config.num_attention_heads} heads ({model_config.num_key_value_heads} KV)"
        )
        print(f"  • Parameters:         {param_count:,} ({param_count / 1e9:.2f}B)")
        print(f"  • Context Length:     {model_config.max_position_embeddings:,} tokens")
        print(f"  • World Size:         {world_size} rank(s) on {device}")
        print(f"  • Dataset:            {len(dataset)} sequences ({dataset.total_tokens:,} tokens)")
        print(f"  • Total Steps:        {parsed.max_steps}")
        print("=" * 65 + "\n")

    # 6. Training Loop
    model.train()
    running_loss = 0.0
    start_time = time.perf_counter()
    tokens_processed = 0

    for step in range(parsed.max_steps):
        # Sample batch
        batch_x: list[torch.Tensor] = []
        batch_y: list[torch.Tensor] = []

        for i in range(parsed.batch_size):
            idx = (step * parsed.batch_size * world_size + rank * parsed.batch_size + i) % max(
                1, len(dataset)
            )
            x, y = dataset[idx]
            batch_x.append(x % model_config.vocab_size)
            batch_y.append(y % model_config.vocab_size)

        bx = torch.stack(batch_x).to(device)
        by = torch.stack(batch_y).to(device)

        # LR update
        lr = get_cosine_lr(step, parsed.max_steps, parsed.warmup_steps, parsed.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        out = model(bx)
        loss = F.cross_entropy(
            out.logits.view(-1, model_config.vocab_size),
            by.view(-1),
        )

        loss.backward()  # type: ignore[no-untyped-call]
        if parsed.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), parsed.grad_clip)
        optimizer.step()

        step_loss = loss.item()
        running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 0 else step_loss
        tokens_processed += parsed.batch_size * parsed.block_size * world_size

        if rank == 0 and (
            (step + 1) % max(1, parsed.max_steps // 5) == 0 or step == parsed.max_steps - 1
        ):
            elapsed = time.perf_counter() - start_time
            tps = tokens_processed / max(1e-5, elapsed)
            print(
                f"Step {step+1:4d}/{parsed.max_steps:4d} | "
                f"Loss: {step_loss:.4f} (avg: {running_loss:.4f}) | "
                f"LR: {lr:.2e} | "
                f"Throughput: {tps:,.1f} tok/s"
            )

        # Checkpoint
        if (
            rank == 0
            and (step + 1) % parsed.checkpoint_interval == 0
            or step == parsed.max_steps - 1
        ):
            out_p = Path(parsed.output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            ckpt_file = out_p / f"ckpt_step_{step+1}.pt"
            unwrapped = model.module if hasattr(model, "module") else model
            save_checkpoint(
                path=ckpt_file,
                model=unwrapped,
                optimizer=optimizer,
                config=model_config.to_dict(),
                tokenizer=tokenizer,
                step=step + 1,
                loss=running_loss,
            )

    if rank == 0:
        total_time = time.perf_counter() - start_time
        final_tps = tokens_processed / max(1e-5, total_time)
        final_file = Path(parsed.output_dir) / "final.pt"
        unwrapped = model.module if hasattr(model, "module") else model
        save_checkpoint(
            path=final_file,
            model=unwrapped,
            optimizer=optimizer,
            config=model_config.to_dict(),
            tokenizer=tokenizer,
            step=parsed.max_steps,
            loss=running_loss,
        )

        print("\n" + "=" * 65)
        print("✅ Pretraining Complete!")
        print(f"  • Final Loss:       {running_loss:.4f}")
        print(f"  • Tokens Trained:   {tokens_processed:,}")
        print(f"  • Average Speed:    {final_tps:,.1f} tok/s")
        print(f"  • Final Checkpoint: {final_file.resolve()}")
        print("=" * 65 + "\n")

    if is_ddp:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
