from __future__ import annotations

from pathlib import Path

from bharat.models.config import BharatModelConfig
from bharat.models.sizing import calculate_parameter_count

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestBharat10BConfig:
    def test_10b_yaml_config_loading(self):
        yaml_p = ROOT_DIR / "configs" / "models" / "bharat-10b.yaml"
        assert yaml_p.is_file(), "bharat-10b.yaml must exist"

        cfg = BharatModelConfig.from_yaml(yaml_p)
        assert cfg.hidden_size == 4096
        assert cfg.intermediate_size == 14336
        assert cfg.num_hidden_layers == 44
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 64000
        assert cfg.rope_theta == 500000.0

    def test_10b_parameter_count(self):
        yaml_p = ROOT_DIR / "configs" / "models" / "bharat-10b.yaml"
        cfg = BharatModelConfig.from_yaml(yaml_p)
        params = calculate_parameter_count(cfg)
        assert 10_000_000_000 <= params.total <= 10_500_000_000
        assert params.total == 10121220096
