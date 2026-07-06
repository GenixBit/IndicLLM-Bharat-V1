from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return ROOT / "configs"


@pytest.fixture
def gpt2_10m_config(configs_dir: Path) -> dict:
    with open(configs_dir / "gpt2-10m.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def gpt2_124m_config(configs_dir: Path) -> dict:
    with open(configs_dir / "gpt2-124m.yaml") as f:
        return yaml.safe_load(f)
