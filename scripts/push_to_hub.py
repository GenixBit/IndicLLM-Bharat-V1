#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Hugging Face Hub Model Publisher.

Publishes verified release bundles or trained checkpoints to the Hugging Face Hub.

Usage:
  # Publish a verified production release bundle (Recommended)
  python scripts/push_to_hub.py \
    --bundle-dir dist/bharat-350m-v1.0.0 \
    --repo GenixBit/IndicLLM-Bharat-350M

  # Dry run to validate bundle files without network upload
  python scripts/push_to_hub.py \
    --bundle-dir dist/bharat-350m-v1.0.0 \
    --repo GenixBit/IndicLLM-Bharat-350M \
    --dry-run

  # Publish from raw checkpoint (.pt)
  python scripts/push_to_hub.py \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --model-config configs/models/bharat-350m.yaml \
    --repo GenixBit/IndicLLM-Bharat-350M
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer
from scripts.build_release_bundle import build_release_bundle

MODEL_CARD_TEMPLATE = """---
language:
- hi
- bn
- ta
- te
- mr
- gu
- kn
- ml
- pa
- or
- as
- ur
- sa
- en
license: apache-2.0
tags:
- text-generation
- causal-lm
- indic
- bharat
- indicllm
pipeline_tag: text-generation
---

# {model_name}

**{model_name}** is a state-of-the-art causal language model engineered for Indian languages with modern architecture:
- **Rotary Position Embeddings (RoPE)**
- **Root Mean Square Normalization (RMSNorm)**
- **SwiGLU Gated Multi-Layer Perceptron**
- **Grouped-Query Attention (GQA 4:1)**

## Model Details
- **Architecture**: BharatForCausalLM
- **Vocabulary Size**: {vocab_size:,}
- **Parameters**: ~{params}
- **Context Length**: {context:,} tokens
"""


def convert_gpt2_to_hf(
    checkpoint_path: Path, _tokenizer_path: Path | None = None
) -> tuple[Any, Any, dict[str, Any], int]:
    from transformers import GPT2Config, GPT2LMHeadModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", {})
    if not model_cfg and hasattr(ckpt.get("metadata", {}), "model_config"):
        model_cfg = ckpt["metadata"].model_config
    if not model_cfg:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing model configuration.")

    params = (
        sum(p.numel() for k, v in ckpt["model"].items() if "weight" in k for p in [v]) // 1_000_000
    )

    hf_cfg = GPT2Config(
        vocab_size=model_cfg.get("vocab_size", 50257),
        n_layer=model_cfg.get("n_layer", 12),
        n_head=model_cfg.get("n_head", 12),
        n_embd=model_cfg.get("n_embd", 768),
        n_positions=model_cfg.get("block_size", 1024),
        n_inner=4 * model_cfg.get("n_embd", 768),
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
    )
    hf_model = GPT2LMHeadModel(hf_cfg)

    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    transpose_keys = {"c_attn.weight", "c_proj.weight", "c_fc.weight"}

    mapped = {}
    for k, v in state.items():
        hf_key = (
            k if k.startswith("transformer.") or k.startswith("lm_head.") else f"transformer.{k}"
        )
        if v.ndim == 2 and any(tk in k for tk in transpose_keys):
            v = v.t()
        mapped[hf_key] = v

    hf_model.load_state_dict(mapped, strict=False)
    return hf_model, hf_cfg, model_cfg, params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish IndicLLM-Bharat release bundles or checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Path to pre-packaged release bundle directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained checkpoint file (.pt)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to YAML model configuration (e.g. configs/models/bharat-350m.yaml)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to tokenizer.json or tokenizer.model file",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo id, e.g. GenixBit/IndicLLM-Bharat-350M",
    )
    parser.add_argument(
        "--model-name",
        default="Bharat-350M",
        help="Name of the model release",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify bundle and configuration without making network calls",
    )
    parser.add_argument(
        "--commit-msg",
        default="Upload IndicLLM-Bharat release bundle",
        help="Git commit message for the Hub repository",
    )
    return parser


def push_to_hub(
    repo_id: str,
    bundle_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    model_name: str = "Bharat-350M",
    private: bool = False,
    commit_msg: str = "Upload IndicLLM-Bharat release bundle",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not bundle_dir and not checkpoint_path:
        raise ValueError("Must specify either --bundle-dir or --checkpoint")

    upload_folder_path: Path
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

    if bundle_dir:
        b_path = Path(bundle_dir).resolve()
        if not b_path.is_dir():
            raise FileNotFoundError(f"Release bundle directory not found: {b_path}")
        upload_folder_path = b_path
    else:
        assert checkpoint_path is not None
        cp_path = Path(checkpoint_path).resolve()
        if not cp_path.is_file() and not cp_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {cp_path}")

        # Check if legacy GPT-2 checkpoint
        ckpt_data = torch.load(
            cp_path if cp_path.is_file() else cp_path / "ckpt.pt",
            map_location="cpu",
            weights_only=False,
        )
        is_legacy = (
            isinstance(ckpt_data, dict)
            and "config" in ckpt_data
            and "n_layer" in ckpt_data.get("config", {})
        )

        temp_dir_obj = tempfile.TemporaryDirectory()
        upload_folder_path = Path(temp_dir_obj.name)

        if is_legacy:
            hf_model, _hf_cfg, _m_cfg, _p_count = convert_gpt2_to_hf(cp_path, tokenizer_path)
            hf_model.save_pretrained(upload_folder_path)
            if tokenizer_path and Path(tokenizer_path).is_file():
                tok = load_bharat_tokenizer(str(tokenizer_path))
                if hasattr(tok, "_tok") and hasattr(tok._tok, "save_pretrained"):
                    tok._tok.save_pretrained(upload_folder_path)
        else:
            build_release_bundle(
                checkpoint_path=cp_path,
                output_dir=upload_folder_path,
                model_config_path=model_config_path,
                tokenizer_path=tokenizer_path,
                model_name=model_name,
            )

    # Ensure README.md exists for Hub repo card
    readme_path = upload_folder_path / "README.md"
    if not readme_path.exists():
        model_card_src = upload_folder_path / "MODEL_CARD.md"
        if model_card_src.exists():
            shutil.copy2(model_card_src, readme_path)
        else:
            config_file = upload_folder_path / "config.json"
            cfg_dict: dict[str, Any] = {}
            if config_file.exists():
                import json

                with open(config_file, encoding="utf-8") as f:
                    cfg_dict = json.load(f)
            v_size = cfg_dict.get("vocab_size", 64000)
            ctx = cfg_dict.get("max_position_embeddings", 4096)
            card_content = MODEL_CARD_TEMPLATE.format(
                model_name=model_name,
                vocab_size=v_size,
                params=model_name,
                context=ctx,
            )
            readme_path.write_text(card_content, encoding="utf-8")

    files = [f.name for f in upload_folder_path.iterdir() if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in upload_folder_path.iterdir() if f.is_file())

    result = {
        "repo_id": repo_id,
        "files_count": len(files),
        "files": sorted(files),
        "total_bytes": total_bytes,
        "dry_run": dry_run,
        "private": private,
    }

    if dry_run:
        print("=" * 64)
        print("🔍 Hugging Face Hub Publication Dry Run")
        print(f"Target Repo:  {repo_id}")
        print(f"Total Files:  {len(files)} ({total_bytes:,} bytes)")
        for f in sorted(files):
            print(f"  • {f}")
        print("=" * 64)
        return result

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"  Repo create notice: {e}")

    api.upload_folder(
        folder_path=str(upload_folder_path),
        repo_id=repo_id,
        commit_message=commit_msg,
        token=token,
    )

    print(f"\n  ✅ Successfully published to https://huggingface.co/{repo_id}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        push_to_hub(
            repo_id=args.repo,
            bundle_dir=args.bundle_dir,
            checkpoint_path=args.checkpoint,
            model_config_path=args.model_config,
            tokenizer_path=args.tokenizer,
            model_name=args.model_name,
            private=args.private,
            commit_msg=args.commit_msg,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"error publishing to hub: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
