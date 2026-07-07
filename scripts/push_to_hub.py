#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — HuggingFace Hub Pusher

Converts a trained checkpoint to HuggingFace format and pushes
to the Hub so anyone can use it with `from_pretrained()`.

Usage:
  # Push 124M checkpoint
  python scripts/push_to_hub.py \
    --checkpoint checkpoints/gpt2-124m-indic/ckpt.pt \
    --repo GenixBit/IndicLLM-Bharat-124M \
    --tokenizer data/indic/tokenizer.model

  # Private repo
  python scripts/push_to_hub.py \
    --checkpoint checkpoints/gpt2-124m/ckpt.pt \
    --repo GenixBit/IndicLLM-Bharat-124M \
    --private

Requirements:
  pip install huggingface-hub transformers sentencepiece
  huggingface-cli login  (or set HF_TOKEN env var)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer

MODEL_CARD_TEMPLATE = """---
language:
{lang_yaml}
license: apache-2.0
tags:
- text-generation
- causal-lm
- indic
- bharat
- indicllm
datasets:
- ai4bharat/sangraha
- statmt/cc100
metrics:
- perplexity
---

# IndicLLM-Bharat {size}

**IndicLLM-Bharat** is a family of open-source causal language models pretrained on
high-quality multilingual Indic text. This is the **{size}** variant.

## Supported Languages

{lang_table}

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 style Transformer |
| Parameters | {params}M |
| Layers | {n_layer} |
| Hidden dim | {n_embd} |
| Attention heads | {n_head} |
| Context length | {block_size} tokens |
| Vocabulary | {vocab_size:,} (SentencePiece BPE) |
| Training tokens | {train_tokens}B |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("GenixBit/IndicLLM-Bharat-{size}")
tokenizer = AutoTokenizer.from_pretrained("GenixBit/IndicLLM-Bharat-{size}")

inputs = tokenizer("नमस्ते, आप कैसे हैं?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{{indicllm-bharat-2025,
  title={{IndicLLM-Bharat: Open Indic Language Models}},
  author={{GenixBit Labs}},
  year={{2025}},
  url={{https://github.com/GenixBit/IndicLLM-Bharat-V1}}
}}
```

## License

Apache 2.0 — free for commercial and research use.
"""


def convert_to_hf(ckpt_path: Path, tok_path: Path | None) -> tuple:
    """Load our checkpoint and convert to HuggingFace GPT2LMHeadModel."""
    from transformers import GPT2Config, GPT2LMHeadModel

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    if not model_cfg:
        raise ValueError("Checkpoint missing model config.")

    params = (
        sum(p.numel() for k, v in ckpt["model"].items() if "weight" in k for p in [v]) // 1_000_000
    )

    hf_cfg = GPT2Config(
        vocab_size=model_cfg.get("vocab_size", 50257),
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        n_positions=model_cfg.get("block_size", 1024),
        n_inner=4 * model_cfg["n_embd"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
    )

    hf_model = GPT2LMHeadModel(hf_cfg)

    # Map weights: strip _orig_mod prefix, add transformer.* prefix, transpose Conv1D
    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Keys that need transposing (nn.Linear [out,in] → Conv1D [in,out])
    TRANSPOSE_KEYS = {"c_attn.weight", "c_proj.weight", "c_fc.weight"}

    mapped = {}
    for k, v in state.items():
        hf_key = (
            k if k.startswith("transformer.") or k.startswith("lm_head.") else f"transformer.{k}"
        )
        if v.ndim == 2 and any(tk in k for tk in TRANSPOSE_KEYS):
            v = v.t()
        mapped[hf_key] = v

    missing, _unexpected = hf_model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (may be OK for tied weights)")

    print(f"  Converted: {params}M params")
    return hf_model, hf_cfg, model_cfg, params


def get_tokenizer_for_hub(tok_path: Path | None, vocab_size: int):
    """Return an HF-compatible tokenizer using the unified loader."""
    from transformers import PreTrainedTokenizerFast

    if tok_path and tok_path.exists():
        print(f"  Loading tokenizer from: {tok_path}")
        bharat_tok = load_bharat_tokenizer(str(tok_path))
        # Convert back to HF for hub export
        if hasattr(bharat_tok, "_tok") and isinstance(bharat_tok._tok, PreTrainedTokenizerFast):
            return bharat_tok._tok
        # Fallback: try to load via HF
        try:
            from transformers import LlamaTokenizer

            if tok_path.suffix == ".model":
                return LlamaTokenizer(vocab_file=str(tok_path))
        except Exception:
            pass

    # Fallback: GPT-2 tokenizer
    bharat_tok = load_bharat_tokenizer(None)
    if hasattr(bharat_tok, "_tok") and isinstance(bharat_tok._tok, PreTrainedTokenizerFast):
        return bharat_tok._tok
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="Push IndicLLM to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--repo", required=True, help="HuggingFace repo id, e.g. GenixBit/IndicLLM-Bharat-124M"
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=None, help="Path to tokenizer.model (SentencePiece)"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-msg", default="Upload IndicLLM-Bharat checkpoint")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("  Set HF_TOKEN env var or run: huggingface-cli login")

    # Convert
    hf_model, _hf_cfg, model_cfg, params = convert_to_hf(args.checkpoint, args.tokenizer)
    tokenizer = get_tokenizer_for_hub(args.tokenizer, model_cfg.get("vocab_size", 50257))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Save model + tokenizer locally first
        hf_model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)

        # Write model card
        size = f"{params}M"
        langs = [
            "Hindi (hi)",
            "Bengali (bn)",
            "Tamil (ta)",
            "Telugu (te)",
            "Marathi (mr)",
            "Gujarati (gu)",
            "Kannada (kn)",
            "Malayalam (ml)",
        ]
        lang_yaml = "\n".join(f"- {l.split(' ')[0].lower()}" for l in langs)
        lang_table = "| Language | Script |\n|----------|--------|\n"
        scripts = {
            "Hindi": "Devanagari",
            "Bengali": "Bengali",
            "Tamil": "Tamil",
            "Telugu": "Telugu",
            "Marathi": "Devanagari",
            "Gujarati": "Gujarati",
            "Kannada": "Kannada",
            "Malayalam": "Malayalam",
        }
        for lang in langs:
            name = lang.split(" ")[0]
            lang_table += f"| {lang} | {scripts.get(name, '—')} |\n"

        card = MODEL_CARD_TEMPLATE.format(
            size=size,
            lang_yaml=lang_yaml,
            lang_table=lang_table,
            params=params,
            n_layer=model_cfg["n_layer"],
            n_embd=model_cfg["n_embd"],
            n_head=model_cfg["n_head"],
            block_size=model_cfg.get("block_size", 1024),
            vocab_size=model_cfg.get("vocab_size", 50257),
            train_tokens="~7",
        )
        (tmp_path / "README.md").write_text(card)

        # Push to Hub
        print(f"\n  Pushing to hub: {args.repo}")
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        try:
            api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True)
        except Exception as e:
            print(f"  Repo create: {e}")

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=args.repo,
            commit_message=args.commit_msg,
            token=token,
        )

    print(f"\n  ✅ Pushed to https://huggingface.co/{args.repo}")
    print("\n  Load it with:")
    print("    from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{args.repo}')")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{args.repo}')")


if __name__ == "__main__":
    main()
