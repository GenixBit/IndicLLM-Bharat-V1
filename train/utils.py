"""Shared utilities for llm-lab."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device_preference() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_wandb(config: dict, job_type: str = "train") -> None:
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set; skipping W&B logging.")
        return

    import wandb

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", wandb_cfg.get("project", "llm-lab")),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=wandb_cfg.get("run_name", config.get("name", "run")),
        config=config,
        job_type=job_type,
    )
