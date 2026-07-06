"""End-to-end pretraining smoke tests: train from scratch and resume from checkpoint."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from train.pretrain import train_from_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tiny_shards(tmpdir: Path, vocab_size: int = 128, length: int = 10_000):
    """Write train.bin, val.bin and meta.pkl under *tmpdir* / shards/."""
    shards = tmpdir / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    for name in ("train", "val"):
        arr = rng.randint(0, vocab_size, size=length, dtype=np.uint16)
        (shards / f"{name}.bin").write_bytes(arr.tobytes())

    with open(shards / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size}, f)
    return shards


def _config_for(out_dir: Path, shards: Path, *, max_iters: int = 5) -> dict:
    return {
        "name": "test-pretrain",
        "model": {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 64,
            "vocab_size": 128,
            "bias": False,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_iters": max_iters,
            "eval_interval": 1,  # checkpoint every iter for predictable resume
            "eval_iters": 2,
            "log_interval": 5,
            "learning_rate": 1e-3,
            "min_lr": 1e-4,
            "warmup_iters": 2,
            "lr_decay_iters": max_iters,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "dtype": "float32",
            "compile": False,
        },
        "data": {
            "train_bin": str(shards / "train.bin"),
            "val_bin": str(shards / "val.bin"),
            "meta_pkl": str(shards / "meta.pkl"),
        },
        "checkpoint": {
            "out_dir": str(out_dir),
            "init_from": "scratch",
        },
        "wandb": {"enabled": False},
    }


def _save_resume_ckpt(path: Path, *, model, optimizer, iter_num, rng_state, metadata, config):
    torch.save(
        {
            "model": model,
            "optimizer": optimizer,
            "iter_num": iter_num,
            "config": config,
            "metadata": metadata,
            "rng_state": rng_state,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPretrain:
    def test_train_from_scratch(self, tmp_path: Path):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, max_iters=3)

        result = train_from_config(cfg)
        assert result["completed_steps"] == 3
        assert result["final_loss"] > 0
        assert result["output_dir"] == str(out_dir)

        # Checkpoints written (eval_interval=1 so saved at iters 1,2,3)
        assert (out_dir / "ckpt.pt").exists()
        assert (out_dir / "final.pt").exists()

        # Verify checkpoint metadata — vocab_size is the tokenizer's
        ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ckpt["iter_num"] == 3
        meta = ckpt["metadata"]
        assert isinstance(meta["vocab_size"], int) and meta["vocab_size"] > 0
        assert meta["tokenizer_type"] == "gpt2"  # default fallback
        assert "git_sha" in meta

    def test_resume_from_checkpoint(self, tmp_path: Path):
        """Run 3 iters → save checkpoint → resume → run 2 more → total 5."""
        shards = _create_tiny_shards(tmp_path)

        # Phase 1: train 3 iters
        out_dir_a = tmp_path / "run_a"
        cfg_a = _config_for(out_dir_a, shards, max_iters=3)
        train_from_config(cfg_a)
        ckpt3 = torch.load(out_dir_a / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ckpt3["iter_num"] == 3

        # Phase 2: resume and run 2 more
        out_dir_b = tmp_path / "run_b"
        out_dir_b.mkdir(parents=True, exist_ok=True)
        _save_resume_ckpt(
            out_dir_b / "ckpt.pt",
            model=ckpt3["model"],
            optimizer=ckpt3["optimizer"],
            iter_num=ckpt3["iter_num"],
            rng_state=ckpt3["rng_state"],
            metadata=ckpt3["metadata"],
            config=_config_for(out_dir_b, shards, max_iters=5),
        )
        cfg_b = _config_for(out_dir_b, shards, max_iters=5)
        cfg_b["checkpoint"]["init_from"] = "resume"
        result = train_from_config(cfg_b)

        assert result["completed_steps"] == 5
        assert result["final_loss"] > 0

    def test_resume_loss_stability(self, tmp_path: Path):
        """Fresh 5-iter run and resumed 5-iter run should reach similar loss."""
        shards = _create_tiny_shards(tmp_path)

        # Fresh train 5 iters
        out_dir_a = tmp_path / "run_a"
        cfg_a = _config_for(out_dir_a, shards, max_iters=5)
        train_from_config(cfg_a)
        ckpt = torch.load(out_dir_a / "ckpt.pt", map_location="cpu", weights_only=False)

        # Resume and train 2 more (total 7)
        out_dir_b = tmp_path / "run_b"
        out_dir_b.mkdir(parents=True, exist_ok=True)
        _save_resume_ckpt(
            out_dir_b / "ckpt.pt",
            model=ckpt["model"],
            optimizer=ckpt["optimizer"],
            iter_num=ckpt["iter_num"],
            rng_state=ckpt["rng_state"],
            metadata=ckpt["metadata"],
            config=_config_for(out_dir_b, shards, max_iters=7),
        )
        cfg_b = _config_for(out_dir_b, shards, max_iters=7)
        cfg_b["checkpoint"]["init_from"] = "resume"
        res_b = train_from_config(cfg_b)

        # Fresh train 7 iters from scratch
        out_dir_c = tmp_path / "run_c"
        cfg_c = _config_for(out_dir_c, shards, max_iters=7)
        res_c = train_from_config(cfg_c)

        assert res_b["completed_steps"] == 7
        assert res_c["completed_steps"] == 7
        # Both should reach similar final loss
        assert abs(res_c["final_loss"] - res_b["final_loss"]) < 1.0

    def test_missing_optimizer_raises(self, tmp_path: Path):
        """Resuming without optimizer state raises ValueError."""
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        bad_ckpt = {
            "model": {"dummy": torch.zeros(1)},
            "iter_num": 5,
            "rng_state": {
                "python": __import__("random").getstate(),
                "torch": torch.get_rng_state().tolist(),
            },
        }
        torch.save(bad_ckpt, out_dir / "ckpt.pt")

        cfg = _config_for(out_dir, shards)
        cfg["checkpoint"]["init_from"] = "resume"
        with pytest.raises(ValueError, match="optimizer"):
            train_from_config(cfg)
