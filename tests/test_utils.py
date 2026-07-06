from __future__ import annotations

from pathlib import Path

import pytest

from train.utils import ensure_dir, load_config


def test_load_config_gpt2_10m(gpt2_10m_config: dict) -> None:
    assert "model" in gpt2_10m_config
    model = gpt2_10m_config["model"]
    assert model["n_layer"] == 6
    assert model["n_head"] == 6
    assert model["n_embd"] == 384
    assert model["block_size"] == 512
    assert model["vocab_size"] == 50257


def test_load_config_gpt2_124m(gpt2_124m_config: dict) -> None:
    assert "model" in gpt2_124m_config
    model = gpt2_124m_config["model"]
    assert model["n_layer"] == 12
    assert model["n_head"] == 12
    assert model["n_embd"] == 768
    assert model["block_size"] == 2048
    assert model["vocab_size"] == 50257


def test_ensure_dir_creates(tmp_path: Path) -> None:
    new_dir = tmp_path / "a" / "b" / "c"
    assert not new_dir.exists()
    result = ensure_dir(new_dir)
    assert new_dir.exists()
    assert result == new_dir


def test_ensure_dir_existing(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir(parents=True, exist_ok=True)
    result = ensure_dir(existing)
    assert existing.exists()
    assert result == existing


def test_load_config_missing_file(configs_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(configs_dir / "nonexistent.yaml")
