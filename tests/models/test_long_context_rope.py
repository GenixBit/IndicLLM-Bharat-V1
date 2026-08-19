from __future__ import annotations

import torch

from bharat.models.attention import GroupedQueryAttention
from bharat.models.config import BharatModelConfig
from bharat.models.rotary import RotaryEmbedding, apply_rotary_pos_emb


class TestLongContextRoPE:
    def test_standard_rope_shape_and_values(self):
        rope = RotaryEmbedding(head_dim=64, max_position_embeddings=4096)
        cos, sin = rope(seq_len=128)
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_yarn_rope_scaling(self):
        rope = RotaryEmbedding(
            head_dim=64,
            max_position_embeddings=32768,
            rope_scaling={"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096},
        )
        assert rope.scaling_type == "yarn"
        assert rope.scaling_factor == 8.0
        assert rope.yarn_temp > 1.0

        cos, sin = rope(seq_len=8192)
        assert cos.shape == (8192, 32)
        assert sin.shape == (8192, 32)
        assert not torch.isnan(cos).any()

    def test_dynamic_ntk_rope_scaling(self):
        rope = RotaryEmbedding(
            head_dim=64,
            max_position_embeddings=32768,
            rope_scaling={
                "type": "dynamic_ntk",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            },
        )
        assert rope.scaling_type == "dynamic_ntk"

        # Under base context
        cos1, _ = rope(seq_len=2048)
        # Over base context (triggers dynamic NTK)
        cos2, _ = rope(seq_len=8192)

        assert cos1.shape == (2048, 32)
        assert cos2.shape == (8192, 32)

    def test_linear_rope_scaling(self):
        rope = RotaryEmbedding(
            head_dim=64,
            max_position_embeddings=16384,
            rope_scaling={"type": "linear", "factor": 4.0},
        )
        assert rope.scaling_type == "linear"
        cos, sin = rope(seq_len=1024)
        assert cos.shape == (1024, 32)

    def test_gqa_with_yarn_scaling(self):
        config = BharatModelConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=32768,
            rope_scaling={"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096},
        )

        attn = GroupedQueryAttention(config)
        x = torch.randn(2, 64, config.hidden_size)
        out, cache = attn(x)

        assert out.shape == (2, 64, config.hidden_size)
        assert not torch.isnan(out).any()

    def test_apply_rotary_emb_dimensions(self):
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 2, 32, 64)
        cos = torch.randn(32, 32)
        sin = torch.randn(32, 32)

        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        assert qr.shape == q.shape
        assert kr.shape == k.shape
