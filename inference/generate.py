#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Interactive Text Generation & Chat CLI.

Load a trained checkpoint and generate text interactively in a REPL,
or run single-shot from the command line across BharatForCausalLM and GPT models.

Usage:
  # Single prompt with modern Bharat model
  python inference/generate.py --checkpoint checkpoints/bharat-350m/final.pt \
    --prompt "भारत का इतिहास" --max-tokens 100

  # Interactive REPL
  python inference/generate.py --checkpoint checkpoints/bharat-350m/final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer
from train.pretrain import GPT, GPTConfig


def load_checkpoint(
    ckpt_path: str | Path,
    device: str = "cpu",
    tokenizer_override: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any] | BharatModelConfig, BharatTokenizer]:
    """Load model + tokenizer from checkpoint, supporting both Bharat and legacy architectures."""
    p = Path(ckpt_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")

    print(f"\n  Loading: {p}")
    ckpt = torch.load(p, map_location=device, weights_only=False)

    # 1. Detect if Native Bharat Model
    is_bharat = False
    model_cfg_obj: Any = None

    if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
        model_cfg_dict = ckpt["metadata"].model_config
        model_cfg_obj = BharatModelConfig.from_dict(model_cfg_dict)
        is_bharat = True
    elif "model_config" in ckpt:
        m_cfg = ckpt["model_config"]
        model_cfg_obj = BharatModelConfig.from_dict(
            m_cfg if isinstance(m_cfg, dict) else m_cfg.__dict__
        )
        is_bharat = True
    elif "config" in ckpt and "hidden_size" in ckpt.get("config", {}):
        model_cfg_obj = BharatModelConfig.from_dict(ckpt["config"])
        is_bharat = True
    elif "model" in ckpt:
        # Inspect state keys to detect architecture
        keys = list(ckpt["model"].keys())
        if any("layers." in k for k in keys) or any("model.embed_tokens" in k for k in keys):
            is_bharat = True
            model_cfg_obj = BharatModelConfig()

    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    if is_bharat:
        if model_cfg_obj is None:
            model_cfg_obj = BharatModelConfig()
        model = BharatForCausalLM(model_cfg_obj).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        params = sum(p_t.numel() for p_t in model.parameters()) / 1e6
        block_size = model_cfg_obj.max_position_embeddings
        print("  Architecture: BharatForCausalLM (RoPE, RMSNorm, SwiGLU, GQA)")
        print(f"  Parameters  : {params:.1f}M params")
        print(f"  Context     : {block_size} tokens")
    else:
        # Legacy GPT-2 Architecture
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", ckpt.get("model_args", {}))
        if not model_cfg:
            model_cfg = {
                "vocab_size": 50257,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
            }
        gpt_cfg = GPTConfig(**model_cfg)
        model = GPT(gpt_cfg).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        params = sum(p_t.numel() for p_t in model.parameters()) / 1e6
        block_size = gpt_cfg.block_size
        model_cfg_obj = model_cfg
        print("  Architecture: GPT-2 Legacy")
        print(f"  Parameters  : {params:.1f}M params")
        print(f"  Context     : {block_size} tokens")

    print(f"  Device      : {device.upper()}")

    # Load Tokenizer
    tok_src = tokenizer_override
    if not tok_src and "metadata" in ckpt and hasattr(ckpt["metadata"], "tokenizer_type"):
        tok_src = getattr(ckpt["metadata"], "tokenizer_type", None)
    if not tok_src and "tokenizer" in ckpt.get("config", {}):
        tok_src = ckpt["config"]["tokenizer"].get("source")

    tokenizer = load_bharat_tokenizer(tok_src)
    print(f"  Tokenizer   : {tokenizer.tokenizer_type} (vocab: {tokenizer.vocab_size:,})")

    return model, model_cfg_obj, tokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: BharatTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
    show_speed: bool = True,
) -> str:
    """Generate text from a prompt using top-k + top-p nucleus sampling."""
    block_size = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "block_size", 1024)
    )

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids: list[int] = []
    t0 = time.time()

    for _ in range(max_tokens):
        x_cond = x[:, -block_size:]

        with ctx:
            out = model(x_cond)
            logits = out.logits if hasattr(out, "logits") else out[0]

        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        else:
            next_token = torch.multinomial(probs[0], 1)

        tok_id = int(next_token.item())
        generated_ids.append(tok_id)

        if tok_id == tokenizer.eos_token_id:
            break

        x = torch.cat([x, next_token.view(1, 1)], dim=1)

    dt = time.time() - t0
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if show_speed:
        tok_per_sec = len(generated_ids) / max(dt, 1e-6)
        print(f"\n  [{len(generated_ids)} tokens in {dt:.2f}s = {tok_per_sec:.1f} tok/s]")

    return gen_text


def interactive_repl(
    model: torch.nn.Module,
    tokenizer: BharatTokenizer,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    max_tokens: int = 100,
    device: str = "cpu",
) -> None:
    """Interactive text generation REPL."""
    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat — Interactive Generation REPL")
    print(f"  Temperature: {temperature}  Top-k: {top_k}  Top-p: {top_p}")
    print(f"  Max tokens : {max_tokens}")
    print("  Commands   : /quit  /temp <v>  /topk <v>  /topp <v>  /max <v>")
    print(f"{'=' * 60}\n")

    temp = temperature
    k = top_k
    p = top_p
    max_t = max_tokens

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
            k = int(prompt.split()[1])
            print(f"  Top-k → {k}")
            continue
        if prompt.startswith("/topp "):
            p = float(prompt.split()[1])
            print(f"  Top-p → {p}")
            continue
        if prompt.startswith("/max "):
            max_t = int(prompt.split()[1])
            print(f"  Max tokens → {max_t}")
            continue

        result = generate(model, tokenizer, prompt, max_t, temp, k, p, device)
        print(f"\n  {prompt}{result}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndicLLM-Bharat Interactive Generator")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (skips REPL)")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Path to tokenizer file or model ID"
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default=None, help="cpu/cuda/mps (auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(args.seed)

    try:
        model, _model_cfg, tokenizer = load_checkpoint(
            args.checkpoint,
            device=device,
            tokenizer_override=args.tokenizer,
        )
    except Exception as e:
        print(f"error loading checkpoint: {e}", file=sys.stderr)
        return 1

    if args.prompt:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n{args.prompt}{result}\n")
    else:
        interactive_repl(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            device=device,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
