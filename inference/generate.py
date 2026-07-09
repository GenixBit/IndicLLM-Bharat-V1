#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — Interactive Text Generation CLI

Load a trained checkpoint and generate text interactively in a REPL,
or run single-shot from the command line.

Usage:
  # Interactive REPL
  python inference/generate.py --checkpoint checkpoints/gpt2-10m/ckpt.pt

  # Single prompt
  python inference/generate.py --checkpoint checkpoints/gpt2-10m/ckpt.pt \
    --prompt "भारत एक" --max-tokens 200

  # With sampling params
  python inference/generate.py --checkpoint checkpoints/gpt2-10m/ckpt.pt \
    --temperature 0.8 --top-k 50 --top-p 0.95
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer  # noqa: E402
from train.pretrain import GPT, GPTConfig  # noqa: E402


def load_checkpoint(ckpt_path: Path, device: str):
    """Load model + tokenizer from checkpoint."""
    print(f"\n  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", ckpt.get("model_args", {}))

    if not model_cfg:
        raise ValueError("Checkpoint missing model config section")

    model = GPT(GPTConfig(**model_cfg)).to(device)

    # Handle _orig_mod prefix from torch.compile
    state = ckpt["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        state2 = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state2, strict=True)

    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    block_size = model.config.block_size
    iter_num = ckpt.get("iter_num", "?")

    print(f"  Model  : {params:.1f}M params")
    print(f"  Iter   : {iter_num}")
    print(f"  Context: {block_size} tokens")
    print(f"  Device : {device.upper()}")

    # Load tokenizer from checkpoint metadata or config
    tok_src = None
    meta = ckpt.get("metadata", {})
    if meta.get("tokenizer_type"):
        print(f"  Tokenizer: {meta.get('tokenizer_type')} (from checkpoint metadata)")
    tok_src = cfg.get("tokenizer", {}).get("source")
    tokenizer = load_bharat_tokenizer(tok_src)

    if cfg.get("name"):
        print(f"  Config : {cfg.get('name')}")

    return model, model_cfg, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
    show_speed: bool = True,
) -> str:
    """Generate text from a prompt using top-k + top-p sampling."""
    block_size = model.config.block_size

    # Autocast for CUDA
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = []
    t0 = time.time()

    for _ in range(max_tokens):
        # Truncate to block_size
        x_cond = x[:, -block_size:]

        with ctx:
            logits, _ = model(x_cond)

        # Only last position
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        else:
            next_token = torch.multinomial(probs[0], 1)

        tok_id = next_token.item()
        generated_ids.append(tok_id)

        # Stop at EOS
        if tok_id == tokenizer.eos_token_id:
            break

        x = torch.cat([x, next_token.view(1, 1)], dim=1)

    dt = time.time() - t0
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if show_speed:
        tok_per_sec = len(generated_ids) / max(dt, 1e-6)
        print(f"\n  [{len(generated_ids)} tokens in {dt:.1f}s = {tok_per_sec:.1f} tok/s]")

    return gen_text


def interactive_repl(model, tokenizer, args, device):
    """Interactive generation REPL."""
    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat — Interactive Generation")
    print(f"  Temperature: {args.temperature}  Top-k: {args.top_k}  Top-p: {args.top_p}")
    print(f"  Max tokens : {args.max_tokens}")
    print("  Commands   : /quit  /temp <v>  /topk <v>  /topp <v>  /max <v>")
    print(f"{'=' * 60}\n")

    temp = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens

    while True:
        try:
            prompt = input("  prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!\n")
            break

        if not prompt:
            continue
        if prompt == "/quit":
            print("  Bye!\n")
            break
        if prompt.startswith("/temp "):
            temp = float(prompt.split()[1])
            print(f"  Temperature → {temp}")
            continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1])
            print(f"  Top-k → {top_k}")
            continue
        if prompt.startswith("/topp "):
            top_p = float(prompt.split()[1])
            print(f"  Top-p → {top_p}")
            continue
        if prompt.startswith("/max "):
            max_tokens = int(prompt.split()[1])
            print(f"  Max tokens → {max_tokens}")
            continue

        result = generate(model, tokenizer, prompt, max_tokens, temp, top_k, top_p, device)
        print(f"\n  {prompt}{result}\n")


def main():
    parser = argparse.ArgumentParser(description="IndicLLM Interactive Generator")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to ckpt.pt or final.pt"
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (skips REPL)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default=None, help="cpu/cuda/mps (auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(args.seed)

    model, _model_cfg, tokenizer = load_checkpoint(args.checkpoint, device)

    if args.prompt:
        # Single-shot mode
        result = generate(
            model,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            device,
        )
        print(f"\n  {args.prompt}{result}\n")
    else:
        # Interactive REPL
        interactive_repl(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
