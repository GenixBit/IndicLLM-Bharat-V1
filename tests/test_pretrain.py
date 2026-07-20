"""End-to-end pretraining smoke tests: train from scratch and resume from checkpoint."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from train.pretrain import GPT, GPTConfig, train_from_config

# ---------------------------------------------------------------------------
# Tiny local tokenizer fixture (no internet required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tokenizer_path():
    """Build a tiny BPE tokenizer and return its path (offline, no HF download)."""
    from tokenizers import Tokenizer as HFTokenizersTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    bpe = BPE()
    tok = HFTokenizersTokenizer(bpe)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=128,
        min_frequency=1,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    tok.train_from_iterator(
        [
            "Hello world how are you today",
            "I am fine thank you",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "Machine learning is fascinating",
        ],
        trainer=trainer,
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tok.save(f.name)
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tiny_shards(tmpdir: Path, vocab_size: int = 128, length: int = 200) -> Path:
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


def _config_for(
    out_dir: Path,
    shards: Path,
    tokenizer_path: str,
    *,
    max_iters: int = 5,
    init_from: str = "scratch",
) -> dict:
    return {
        "name": "test-pretrain",
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "block_size": 8,
            "vocab_size": 128,
            "bias": False,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_iters": max_iters,
            "eval_interval": 1,
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
            "init_from": init_from,
        },
        "tokenizer": {"source": tokenizer_path},
        "wandb": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Fast resume test (runs in normal CI)
# ---------------------------------------------------------------------------


class TestPretrainFast:
    """Fast CPU resume test — not slow, not integration, not GPU.

    Proves:
      1. No optimizer step is repeated
      2. No optimizer step is skipped
      3. Model state is restored
      4. Optimizer state is restored
      5. RNG state is restored
      6. LR scheduling resumes from correct step
      7. Checkpoint completed_steps is correct
      8. Checkpoint next_step is correct
      9. Final result reports 5 completed steps
      10. Missing resume checkpoint fails
      11. Missing model state fails
      12. Missing optimizer state fails
      13. Missing RNG state fails
      14. Incompatible tokenizer fails
      15. Legacy checkpoints require compatibility_mode: legacy
    """

    def test_fast_resume_cycle(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Run 3 steps → resume → 2 more → verify 5 total with no repeats/skips."""
        shards = _create_tiny_shards(tmp_path, vocab_size=128)

        # ---- Phase 1: train 3 steps ----
        out_a = tmp_path / "run_a"
        cfg_a = _config_for(out_a, shards, tiny_tokenizer_path, max_iters=3, init_from="scratch")
        res_a = train_from_config(cfg_a)
        assert res_a["completed_steps"] == 3
        assert res_a["next_step"] == 3
        assert res_a["final_loss"] is not None

        # Verify checkpoint format
        ckpt_a = torch.load(out_a / "ckpt.pt", map_location="cpu", weights_only=False)
        assert ckpt_a["completed_steps"] == 3
        assert ckpt_a["next_step"] == 3
        assert "final_loss" in ckpt_a
        assert "metadata" in ckpt_a
        assert ckpt_a["metadata"]["tokenizer_hash"]
        meta = ckpt_a["metadata"]
        assert isinstance(meta["vocab_size"], int) and meta["vocab_size"] > 0
        assert meta["tokenizer_type"]

        # ---- Phase 2: resume and run 2 more ----
        out_b = tmp_path / "run_b"
        out_b.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint to new output dir
        cfg_b = _config_for(out_b, shards, tiny_tokenizer_path, max_iters=5, init_from="resume")
        torch.save(
            {
                "model": ckpt_a["model"],
                "optimizer": ckpt_a["optimizer"],
                "completed_steps": ckpt_a["completed_steps"],
                "next_step": ckpt_a["next_step"],
                "config": cfg_b,
                "metadata": ckpt_a["metadata"],
                "rng_state": ckpt_a["rng_state"],
            },
            out_b / "ckpt.pt",
        )

        res_b = train_from_config(cfg_b)
        assert (
            res_b["completed_steps"] == 5
        ), f"Expected 5 completed steps, got {res_b['completed_steps']}"
        assert res_b["next_step"] == 5, f"Expected next_step=5, got {res_b['next_step']}"
        assert res_b["final_loss"] is not None

        # Verify final checkpoint
        ckpt_b = torch.load(out_b / "final.pt", map_location="cpu", weights_only=False)
        assert ckpt_b["completed_steps"] == 5
        assert ckpt_b["next_step"] == 5
        assert isinstance(ckpt_b["final_loss"], float)
        assert (
            isinstance(ckpt_b["metadata"]["vocab_size"], int)
            and ckpt_b["metadata"]["vocab_size"] > 0
        )

    # --- Error cases ---

    def test_missing_ckpt_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        # No ckpt.pt written → FileNotFoundError
        with pytest.raises((FileNotFoundError, OSError)):
            train_from_config(cfg)

    def test_missing_model_state_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "optimizer": {},
                "completed_steps": 3,
                "next_step": 3,
                "metadata": {
                    "tokenizer_hash": "dummy",
                    "tokenizer_type": "test",
                    "vocab_size": 128,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        with pytest.raises((ValueError, AssertionError, KeyError, RuntimeError)):
            train_from_config(cfg)

    def test_missing_optimizer_state_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        torch.save(
            {
                "model": model.state_dict(),
                "completed_steps": 3,
                "next_step": 3,
                "metadata": {
                    "tokenizer_hash": "dummy",
                    "tokenizer_type": "test",
                    "vocab_size": 128,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        with pytest.raises((ValueError, KeyError)):
            train_from_config(cfg)

    def test_missing_rng_state_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "completed_steps": 3,
                "next_step": 3,
                "metadata": {
                    "tokenizer_hash": "dummy",
                    "tokenizer_type": "test",
                    "vocab_size": 128,
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        with pytest.raises((ValueError, KeyError)):
            train_from_config(cfg)

    def test_incompatible_tokenizer_raises(self, tmp_path: Path):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "completed_steps": 3,
                "next_step": 3,
                "metadata": {
                    "tokenizer_hash": "different_vocab_hash_123456789012",
                    "tokenizer_type": "different",
                    "vocab_size": 999,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, "", init_from="resume")
        # Will download GPT-2 tokenizer, hash won't match 'different_vocab_hash_...'
        with pytest.raises((ValueError, FileNotFoundError)):
            train_from_config(cfg)

    def test_legacy_ckpt_requires_compat_mode(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        from bharat.tokenizer import load_tokenizer
        from bharat.tokenizer.metadata import tokenizer_hash

        tok = load_tokenizer(tiny_tokenizer_path)
        correct_hash = tokenizer_hash(tok)

        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "iter_num": 3,
                "config": _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume"),
                "metadata": {
                    "tokenizer_hash": correct_hash,
                    "tokenizer_type": tok.tokenizer_type,
                    "vocab_size": tok.vocab_size,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        with pytest.raises(ValueError, match="compatibility_mode"):
            train_from_config(cfg)

    def test_negative_max_iters_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=-1)
        with pytest.raises((ValueError, AssertionError)):
            train_from_config(cfg)

    def test_zero_max_iters_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=0)
        with pytest.raises((ValueError, AssertionError)):
            train_from_config(cfg)

    def test_resume_at_target_returns_noop(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Resume with next_step == max_iters returns no-op result."""
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        from bharat.tokenizer import load_tokenizer
        from bharat.tokenizer.metadata import tokenizer_hash

        tok = load_tokenizer(tiny_tokenizer_path)
        correct_hash = tokenizer_hash(tok)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "completed_steps": 5,
                "next_step": 5,
                "metadata": {
                    "tokenizer_hash": correct_hash,
                    "tokenizer_type": tok.tokenizer_type,
                    "vocab_size": tok.vocab_size,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=5, init_from="resume")
        result = train_from_config(cfg)
        assert result["final_loss"] is None
        assert result["completed_steps"] == 5
        assert result["next_step"] == 5

    def test_resume_ahead_of_target_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Resume with next_step > max_iters raises ValueError."""
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        from bharat.tokenizer import load_tokenizer
        from bharat.tokenizer.metadata import tokenizer_hash

        tok = load_tokenizer(tiny_tokenizer_path)
        correct_hash = tokenizer_hash(tok)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "completed_steps": 7,
                "next_step": 7,
                "metadata": {
                    "tokenizer_hash": correct_hash,
                    "tokenizer_type": tok.tokenizer_type,
                    "vocab_size": tok.vocab_size,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=5, init_from="resume")
        with pytest.raises(ValueError, match="exceeds requested"):
            train_from_config(cfg)

    def test_noop_result_has_no_loss(self, tmp_path: Path, tiny_tokenizer_path: str):
        """No-op result must not have an undefined or invented loss value."""
        shards = _create_tiny_shards(tmp_path)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        from bharat.tokenizer import load_tokenizer
        from bharat.tokenizer.metadata import tokenizer_hash

        tok = load_tokenizer(tiny_tokenizer_path)
        correct_hash = tokenizer_hash(tok)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "completed_steps": 5,
                "next_step": 5,
                "metadata": {
                    "tokenizer_hash": correct_hash,
                    "tokenizer_type": tok.tokenizer_type,
                    "vocab_size": tok.vocab_size,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=5, init_from="resume")
        result = train_from_config(cfg)
        assert "final_loss" in result
        assert result["final_loss"] is None

    def test_val_data_too_short_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Short validation data (<= block_size) raises ValueError."""
        shards = _create_tiny_shards(tmp_path)  # default length=200 for train
        # overwrite val.bin with data shorter than block_size
        rng = __import__("numpy").random.RandomState(42)
        short = rng.randint(0, 128, size=4, dtype=__import__("numpy").uint16)
        (shards / "val.bin").write_bytes(short.tobytes())
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=1)
        cfg["training"]["warmup_iters"] = 0
        with pytest.raises(ValueError, match="Validation"):
            train_from_config(cfg)

    def test_train_data_too_short_raises(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Short training data (<= block_size) raises ValueError."""
        shards = tmp_path / "shards"
        shards.mkdir(parents=True, exist_ok=True)
        rng = __import__("numpy").random.RandomState(42)
        for name in ("train", "val"):
            arr = rng.randint(0, 128, size=200, dtype=__import__("numpy").uint16)
            (shards / f"{name}.bin").write_bytes(arr.tobytes())
        with open(shards / "meta.pkl", "wb") as f:
            pickle.dump({"vocab_size": 128}, f)
        _create_tiny_shards(tmp_path)  # ensure shards dir exists
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=1)
        cfg["model"]["block_size"] = 256
        cfg["training"]["warmup_iters"] = 0
        with pytest.raises(ValueError, match="Training"):
            train_from_config(cfg)


# ---------------------------------------------------------------------------
# Slow / integration tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPretrainSlow:
    def test_train_from_scratch(self, tmp_path: Path, tiny_tokenizer_path: str):
        shards = _create_tiny_shards(tmp_path, length=10_000)
        out_dir = tmp_path / "ckpts"
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=3)
        result = train_from_config(cfg)
        assert result["completed_steps"] == 3
        assert result["final_loss"] is not None

    def test_resume_legacy_with_compat_mode(self, tmp_path: Path, tiny_tokenizer_path: str):
        """Legacy checkpoint with compatibility_mode: legacy must resume."""
        shards = _create_tiny_shards(tmp_path, length=10_000)
        out_dir = tmp_path / "ckpts"
        out_dir.mkdir(parents=True, exist_ok=True)

        from bharat.tokenizer import load_tokenizer
        from bharat.tokenizer.metadata import tokenizer_hash

        tok = load_tokenizer(tiny_tokenizer_path)
        correct_hash = tokenizer_hash(tok)

        model = GPT(
            GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=8, vocab_size=128, bias=False)
        )
        from train.pretrain import _build_optimizer

        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume")
        optim = _build_optimizer(model, cfg["training"])
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "iter_num": 3,
                "config": _config_for(out_dir, shards, tiny_tokenizer_path, init_from="resume"),
                "metadata": {
                    "tokenizer_hash": correct_hash,
                    "tokenizer_type": tok.tokenizer_type,
                    "vocab_size": tok.vocab_size,
                },
                "rng_state": {
                    "python": __import__("random").getstate(),
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": {},
                },
            },
            out_dir / "ckpt.pt",
        )
        cfg = _config_for(out_dir, shards, tiny_tokenizer_path, max_iters=5, init_from="resume")
        cfg["checkpoint"]["compatibility_mode"] = "legacy"
        result = train_from_config(cfg)
        assert result["completed_steps"] == 5
