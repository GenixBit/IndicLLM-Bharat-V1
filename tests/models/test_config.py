from __future__ import annotations

import pytest

from bharat.models.config import BharatModelConfig


class TestBharatModelConfig:
    def test_valid_config(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        assert cfg.vocab_size == 32000
        assert cfg.head_dim == 128
        assert cfg.num_key_value_groups == 4

    def test_head_dim_property(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        assert cfg.head_dim == 4096 // 32

    def test_num_key_value_groups_property(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        assert cfg.num_key_value_groups == 4

    def test_default_values(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=512,
        )
        assert cfg.rope_theta == 10_000.0
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.attention_dropout == 0.0
        assert cfg.hidden_dropout == 0.0
        assert cfg.initializer_range == 0.02
        assert cfg.attention_bias is False
        assert cfg.mlp_bias is False
        assert cfg.tie_word_embeddings is True

    @pytest.mark.parametrize("field", ["vocab_size", "hidden_size", "intermediate_size"])
    def test_zero_or_negative_dimension_raises(self, field):
        kwargs = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 2048,
        }
        for val in [0, -1]:
            kwargs[field] = val
            with pytest.raises(ValueError, match="must be positive"):
                BharatModelConfig(**kwargs)
            kwargs[field] = 32000  # reset

    def test_hidden_size_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4095,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=2048,
            )

    def test_heads_divisible_by_kv_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=7,
                max_position_embeddings=2048,
            )

    def test_kv_heads_not_exceed_attn_heads(self):
        with pytest.raises(ValueError, match="must not exceed"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=8,
                num_key_value_heads=32,
                max_position_embeddings=2048,
            )

    def test_head_dim_even(self):
        # head_dim = 72 / 8 = 9 (odd), hidden_size divisible by num_attention_heads
        with pytest.raises(ValueError, match="even"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=72,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=8,
                num_key_value_heads=1,
                max_position_embeddings=2048,
            )

    def test_dropout_range(self):
        with pytest.raises(ValueError, match="dropout"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=2048,
                attention_dropout=-0.1,
            )

    def test_rope_theta_positive(self):
        with pytest.raises(ValueError, match="rope_theta"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=2048,
                rope_theta=0.0,
            )

    def test_rms_norm_eps_positive(self):
        with pytest.raises(ValueError, match="rms_norm_eps"):
            BharatModelConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=2048,
                rms_norm_eps=0.0,
            )

    def test_to_dict(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        d = cfg.to_dict()
        assert d["vocab_size"] == 32000
        assert d["hidden_size"] == 4096

    def test_from_dict(self):
        cfg = BharatModelConfig.from_dict(
            {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 2048,
            }
        )
        assert cfg.vocab_size == 32000
        assert cfg.head_dim == 128

    def test_frozen(self):
        cfg = BharatModelConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.vocab_size = 64000
