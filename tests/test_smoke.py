from __future__ import annotations


def test_import_pretrain() -> None:
    from train.pretrain import GPT, GPTConfig
    assert GPT is not None
    assert GPTConfig is not None


def test_import_utils() -> None:
    from train.utils import ensure_dir, get_device_preference, init_wandb, load_config
    assert load_config is not None
    assert get_device_preference is not None
    assert ensure_dir is not None
    assert init_wandb is not None


def test_import_sft() -> None:
    from train.sft import main as sft_main
    assert sft_main is not None


def test_import_dpo() -> None:
    from train.dpo import dpo_loss, log_probs
    from train.dpo import main as dpo_main
    assert dpo_main is not None
    assert dpo_loss is not None
    assert log_probs is not None


def test_import_eval() -> None:
    from eval.benchmark import main as eval_main
    assert eval_main is not None


def test_import_inference() -> None:
    from inference.generate import generate, load_checkpoint, main
    assert generate is not None
    assert load_checkpoint is not None
    assert main is not None


def test_import_api() -> None:
    from inference.api import app
    assert app is not None
