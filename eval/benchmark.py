#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — Model Evaluation & Benchmarking

Evaluates a trained checkpoint on:
  1. Perplexity on held-out validation data
  2. Token-level accuracy
  3. Sample generations (qualitative)
  4. Indic language detection (% of generated text in target script)

Usage:
  python eval/benchmark.py --checkpoint checkpoints/gpt2-10m/ckpt.pt
  python eval/benchmark.py --checkpoint checkpoints/gpt2-124m/final.pt --config configs/gpt2-124m.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT, GPTConfig
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer

INDIC_UNICODE_RANGES = {
    "hi": (0x0900, 0x097F, "Devanagari"),
    "bn": (0x0980, 0x09FF, "Bengali"),
    "ta": (0x0B80, 0x0BFF, "Tamil"),
    "te": (0x0C00, 0x0C7F, "Telugu"),
    "mr": (0x0900, 0x097F, "Devanagari"),
    "gu": (0x0A80, 0x0AFF, "Gujarati"),
    "kn": (0x0C80, 0x0CFF, "Kannada"),
    "ml": (0x0D00, 0x0D7F, "Malayalam"),
}


def load_model(ckpt_path: Path, device: str):
    """Load checkpoint and return model + config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    model = GPT(GPTConfig(**model_cfg)).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    return model, cfg, params


@torch.no_grad()
def eval_perplexity(
    model, data_arr, block_size: int, batch_size: int, eval_iters: int, device: str
) -> dict:
    """Compute perplexity and token accuracy on data."""
    from contextlib import nullcontext

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    losses = []
    correct = 0
    total = 0

    for k in range(eval_iters):
        ix = torch.randint(len(data_arr) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data_arr[i : i + block_size].astype(np.int64)) for i in ix]
        ).to(device)
        y = torch.stack(
            [torch.from_numpy(data_arr[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
        ).to(device)

        with ctx:
            logits, loss = model(x, y)

        losses.append(loss.item())

        # Token accuracy
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    accuracy = correct / max(total, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": accuracy,
        "eval_iters": eval_iters,
    }


@torch.no_grad()
def generate_samples(
    model, tokenizer, prompts: list[str], max_tokens: int, device: str, temperature: float = 0.8
) -> list[dict]:
    """Generate text samples for qualitative evaluation."""
    from contextlib import nullcontext

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    block_size = model.config.block_size
    results = []

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        gen_ids = []
        t0 = time.time()

        for _ in range(max_tokens):
            x_cond = x[:, -block_size:]
            with ctx:
                logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs[0], 1)
            tok_id = next_tok.item()
            gen_ids.append(tok_id)
            if tok_id == tokenizer.eos_token_id:
                break
            x = torch.cat([x, next_tok.view(1, 1)], dim=1)

        dt = time.time() - t0
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Detect Indic scripts in output
        script_counts = {}
        for name, (lo, hi, script) in INDIC_UNICODE_RANGES.items():
            count = sum(1 for c in text if lo <= ord(c) <= hi)
            if count > 0:
                script_counts[script] = script_counts.get(script, 0) + count

        results.append(
            {
                "prompt": prompt,
                "generation": text,
                "tokens": len(gen_ids),
                "time_s": round(dt, 2),
                "tok_per_sec": round(len(gen_ids) / max(dt, 1e-6), 1),
                "scripts_detected": script_counts,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="IndicLLM Benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gen-tokens", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\n{'='*60}")
    print("  IndicLLM-Bharat — Model Benchmark")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device    : {device.upper()}")
    print(f"{'='*60}")

    # Load model
    model, cfg, params = load_model(args.checkpoint, device)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    block_size = model_cfg.get("block_size", 1024)
    iter_num = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
        "iter_num", "?"
    )

    print(f"\n  Model: {params:.1f}M params  |  iter {iter_num}")
    print(
        f"  Block: {block_size}  |  Layers: {model_cfg.get('n_layer')}  |  Heads: {model_cfg.get('n_head')}"
    )

    # ── Perplexity Evaluation ──
    results = {"model": str(args.checkpoint), "params_m": params, "iter": iter_num}

    val_path = Path(data_cfg.get("val_bin", "data/shards/val.bin"))
    train_path = Path(data_cfg.get("train_bin", "data/shards/train.bin"))

    if val_path.exists():
        print(f"\n  Evaluating perplexity on: {val_path.name}")
        val_arr = np.memmap(str(val_path), dtype=np.uint16, mode="r")
        val_metrics = eval_perplexity(
            model,
            val_arr,
            block_size,
            args.batch_size,
            min(args.eval_iters, len(val_arr) // (block_size * args.batch_size)),
            device,
        )
        results["val"] = val_metrics
        print(f"  Val loss      : {val_metrics['loss']:.4f}")
        print(f"  Val perplexity: {val_metrics['perplexity']:.2f}")
        print(f"  Val accuracy  : {val_metrics['token_accuracy']:.4f}")
    else:
        print(f"  ⚠ Val data not found: {val_path}")

    if train_path.exists():
        print(f"\n  Evaluating on: {train_path.name}")
        train_arr = np.memmap(str(train_path), dtype=np.uint16, mode="r")
        train_metrics = eval_perplexity(
            model,
            train_arr,
            block_size,
            args.batch_size,
            min(args.eval_iters, len(train_arr) // (block_size * args.batch_size)),
            device,
        )
        results["train"] = train_metrics
        print(f"  Train loss    : {train_metrics['loss']:.4f}")
        print(f"  Train ppl     : {train_metrics['perplexity']:.2f}")
        gap = abs(val_metrics["loss"] - train_metrics["loss"]) if "val" in results else 0
        print(f"  Overfitting   : {'⚠ YES' if gap > 0.5 else '✓ No'} (gap={gap:.3f})")

    # ── Sample Generations ──
    print("\n  Generating samples...")
    # Load tokenizer from config or default to GPT-2
    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = load_bharat_tokenizer(tok_src)

    prompts = [
        "The capital of India is",
        "In machine learning, the",
        "Once upon a time in a village",
        "The most important",
        "Scientists have discovered that",
    ]

    # Add Indic prompts if model trained on Indic data
    indic_path = Path("data/indic/train.bin")
    if indic_path.exists():
        prompts.extend(
            [
                "भारत एक",
                "தமிழ்நாடு",
                "বাংলাদেশ",
            ]
        )

    samples = generate_samples(model, tokenizer, prompts, args.gen_tokens, device)
    results["samples"] = samples

    print(f"\n  {'─'*56}")
    for s in samples:
        print(f"  Prompt: {s['prompt']}")
        gen_preview = s["generation"][:120].replace("\n", " ")
        print(f"  Output: {gen_preview}...")
        scripts = s.get("scripts_detected", {})
        if scripts:
            print(f"  Scripts: {', '.join(f'{k}({v})' for k, v in scripts.items())}")
        print(f"  [{s['tokens']} tok, {s['tok_per_sec']} tok/s]")
        print(f"  {'─'*56}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"  Model     : {params:.1f}M params ({args.checkpoint.name})")
    if "val" in results:
        print(f"  Val PPL   : {results['val']['perplexity']:.2f}")
        print(f"  Val Acc   : {results['val']['token_accuracy']:.2%}")
    if "train" in results:
        print(f"  Train PPL : {results['train']['perplexity']:.2f}")
    print(f"  Samples   : {len(samples)} prompts generated")
    print(f"{'='*60}\n")

    # Save results
    out_file = args.output or Path(f"eval/results_{args.checkpoint.parent.name}_{iter_num}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean up non-serializable items
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_file}\n")


if __name__ == "__main__":
    main()
