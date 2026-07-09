#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — W&B + lm-eval evaluation tracker.

Loads a checkpoint, computes perplexity on the val shard,
optionally runs lm-eval-harness on key benchmarks, and logs
everything to Weights & Biases.

Usage:
  # Perplexity only (fast, works on CPU/MPS):
  python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt

  # Full benchmark suite:
  python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt \
    --tasks hellaswag,piqa,winogrande,lambada_openai

  # No W&B, just print results:
  python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt --no-wandb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT, GPTConfig  # noqa: E402

BENCHMARK_TASKS = ["hellaswag", "piqa", "winogrande", "lambada_openai"]


# ── Device ───────────────────────────────────────────────────
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Load checkpoint ──────────────────────────────────────────
def load_checkpoint(ckpt_path: Path, device: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", ckpt.get("model_args", {}))
    if not model_cfg:
        raise ValueError("Checkpoint has no model config. Pass --config manually.")
    model = GPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    iter_num = ckpt.get("iter_num", 0)
    print(f"  Iter: {iter_num} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, cfg, iter_num


# ── Perplexity on val shard ──────────────────────────────────
@torch.no_grad()
def compute_perplexity(
    model, val_bin: Path, block_size: int, batch_size: int, eval_iters: int, device: str
) -> float:
    ctx = (
        nullcontext()
        if device in ("cpu", "mps")
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )
    data = np.memmap(str(val_bin), dtype=np.uint16, mode="r")
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
        ).to(device)
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        ).to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


# ── Export HF stub for lm-eval ───────────────────────────────
def export_hf_stub(ckpt_path: Path, model_cfg: dict, export_dir: Path) -> Path:
    from transformers import GPT2Config, GPT2LMHeadModel

    export_dir.mkdir(parents=True, exist_ok=True)
    hf_cfg = GPT2Config(
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        n_positions=model_cfg.get("block_size", 1024),
        vocab_size=model_cfg.get("vocab_size", 50257),
    )
    # Load weights into HF model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hf_model = GPT2LMHeadModel(hf_cfg)
    # Map our weight keys → HF keys
    state = ckpt["model"]
    hf_state = {}
    for k, v in state.items():
        k2 = k.replace("transformer.", "")
        hf_state[k2] = v
    from contextlib import suppress

    with suppress(Exception):
        hf_model.load_state_dict(hf_state, strict=False)
    hf_model.save_pretrained(str(export_dir))
    print(f"  Exported HF stub → {export_dir}")
    return export_dir


# ── Run lm-eval benchmarks ───────────────────────────────────
def run_benchmarks(export_dir: Path, tasks: list[str]) -> dict:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  lm-eval not installed. Skipping benchmarks.")
        print("  Install with: pip install lm-eval")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Running lm-eval on: {tasks}")
    lm = HFLM(pretrained=str(export_dir), device=device, batch_size=8)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, batch_size=8)
    return results.get("results", {})


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IndicLLM eval + W&B tracking")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--val-bin", type=Path, default=Path("data/shards/val.bin"))
    parser.add_argument(
        "--tasks", default=None, help="Comma-separated lm-eval tasks. Skip for perplexity only."
    )
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--export-dir", type=Path, default=Path("checkpoints/hf_export"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    device = pick_device()
    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat-V1 — Evaluation")
    print(f"  Device     : {device.upper()}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"{'=' * 60}\n")

    # Load model
    model, cfg, iter_num = load_checkpoint(args.checkpoint, device)
    model_cfg = cfg.get("model", {})
    block_size = model_cfg.get("block_size", 1024)

    # W&B init
    wandb = None
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb as _wandb

            run_name = args.run_name or f"eval-iter{iter_num}"
            _wandb.init(
                project=cfg.get("wandb", {}).get("project", "indicllm-bharat"),
                name=run_name,
                config={"iter_num": iter_num, "checkpoint": str(args.checkpoint)},
                job_type="eval",
            )
            wandb = _wandb
            print(f"  W&B run: {run_name}\n")
        except Exception as e:
            print(f"  W&B init failed: {e} — continuing without logging")

    results = {"iter_num": iter_num, "checkpoint": str(args.checkpoint)}

    # ── Perplexity ──────────────────────────────────────────
    print(f"[1] Computing perplexity on val shard ({args.eval_iters} batches)...")
    if args.val_bin.exists():
        ppl = compute_perplexity(
            model, args.val_bin, block_size, args.batch_size, args.eval_iters, device
        )
        results["val_perplexity"] = ppl
        results["val_bpb"] = math.log2(ppl)
        print(f"    Perplexity : {ppl:.2f}")
        print(f"    BPB        : {results['val_bpb']:.3f}")
    else:
        print(f"    val.bin not found at {args.val_bin} — skipping perplexity")

    # ── Benchmarks ──────────────────────────────────────────
    if args.tasks:
        print(f"\n[2] Running lm-eval benchmarks: {args.tasks}")
        export_dir = export_hf_stub(args.checkpoint, model_cfg, args.export_dir)
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        bench = run_benchmarks(export_dir, tasks)
        results["benchmarks"] = bench
        if bench:
            print("\n  Results:")
            for task, r in bench.items():
                acc = r.get("acc,none") or r.get("acc_norm,none") or r.get("acc")
                if acc is not None:
                    print(f"    {task:<20}: {acc * 100:.1f}%")
                    results[f"bench/{task}"] = acc

    # ── Save + W&B log ───────────────────────────────────────
    out_file = Path("eval/results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_file}")

    if wandb:
        wandb.log(results, step=iter_num)
        wandb.finish()
        print("  Logged to W&B ✅")

    print(f"\n{'=' * 60}")
    print("  Eval complete.")
    print(
        f"  Perplexity : {results.get('val_perplexity', 'N/A'):.2f}"
        if "val_perplexity" in results
        else "  Perplexity : N/A"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
