#!/usr/bin/env python3
"""
Run lm-eval-harness benchmarks and log results to W&B.

Usage:
  python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt
  python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt --tasks hellaswag,piqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT
from train.utils import init_wandb, load_config


DEFAULT_TASKS = ["hellaswag", "piqa", "winogrande"]


def load_model_from_checkpoint(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Checkpoint missing model config; pass --config explicitly.")
    model = GPT.from_config(model_cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def export_hf_stub(checkpoint: Path, out_dir: Path, config_path: Path | None) -> Path:
    """Export minimal HF-compatible weights for lm-eval."""
    from transformers import GPT2Config, GPT2LMHeadModel

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or load_config(config_path)
    m = cfg["model"]
    hf_cfg = GPT2Config(
        n_layer=m["n_layer"],
        n_head=m["n_head"],
        n_embd=m["n_embd"],
        n_positions=m["block_size"],
        vocab_size=m["vocab_size"],
    )
    model = GPT2LMHeadModel(hf_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


def run_lm_eval(model_path: Path, tasks: list[str]) -> dict:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise SystemExit("Install lm-eval: pip install lm-eval") from e

    lm = HFLM(pretrained=str(model_path), device="cuda" if torch.cuda.is_available() else "cpu")
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, batch_size=8)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--export-dir", type=Path, default=Path("checkpoints/hf_export"))
    args = parser.parse_args()

    cfg_path = args.config or Path("configs/gpt2-124m.yaml")
    cfg = load_config(cfg_path) if cfg_path.exists() else {"wandb": {"enabled": True, "run_name": "eval"}}
    init_wandb({**cfg, "name": "eval"}, job_type="eval")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    export_dir = export_hf_stub(args.checkpoint, args.export_dir, cfg_path)
    print(f"Exported HF stub to {export_dir}")

    results = run_lm_eval(export_dir, tasks)
    out_file = Path("eval/results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results.get("results", results), indent=2))

    import os

    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.log({"eval": results.get("results", {})})


if __name__ == "__main__":
    main()
