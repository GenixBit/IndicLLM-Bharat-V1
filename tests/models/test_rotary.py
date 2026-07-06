from __future__ import annotations

import pytest
import torch

from bharat.models.rotary import RotaryEmbedding, apply_rotary_pos_emb


class TestRotaryEmbedding:
    @pytest.fixture
    def rotary(self):
        return RotaryEmbedding(head_dim=64, max_position_embeddings=2048, rope_theta=10_000.0)

    def test_position_zero_behavior(self, rotary: RotaryEmbedding):
        """At position 0, cos=1 and sin=0 → no rotation."""
        cos, sin = rotary(seq_len=1)
        assert torch.allclose(cos[0], torch.ones_like(cos[0]), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros_like(sin[0]), atol=1e-6)

    def test_output_shape(self, rotary: RotaryEmbedding):
        cos, sin = rotary(seq_len=10)
        assert cos.shape == (10, 32)
        assert sin.shape == (10, 32)

    def test_preserves_vector_norms(self, rotary: RotaryEmbedding):
        q = torch.randn(2, 8, 10, 64)
        q_norm_before = q.norm(dim=-1)
        cos, sin = rotary(seq_len=10)
        q_rotated, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
        q_norm_after = q_rotated.norm(dim=-1)
        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)

    def test_different_positions_differ(self, rotary: RotaryEmbedding):
        cos_1, _sin_1 = rotary(seq_len=1)
        cos_5, _sin_5 = rotary(seq_len=1, offset=4)
        assert not torch.allclose(cos_1, cos_5, atol=1e-6)

    def test_q_and_k_same_rotation(self, rotary: RotaryEmbedding):
        q = torch.randn(2, 8, 5, 64)
        k = torch.randn(2, 8, 5, 64)
        cos, sin = rotary(seq_len=5)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_gradient_flow(self, rotary: RotaryEmbedding):
        q = torch.randn(2, 4, 8, 64, requires_grad=True)
        k = torch.randn(2, 4, 8, 64, requires_grad=True)
        cos, sin = rotary(seq_len=8)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        loss = (q_rot + k_rot).sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert torch.isfinite(q.grad).all()
        assert torch.isfinite(k.grad).all()

    def test_cache_extension(self):
        """RoPE can compute beyond initial max_position_embeddings."""
        rotary = RotaryEmbedding(head_dim=32, max_position_embeddings=16, rope_theta=10_000.0)
        cos_short, _sin_short = rotary(seq_len=16)
        cos_long, sin_long = rotary(seq_len=24)
        assert cos_long.shape == (24, 16)
        assert sin_long.shape == (24, 16)
        assert torch.allclose(cos_long[:16], cos_short, atol=1e-6)

    def test_float32(self):
        rotary = RotaryEmbedding(head_dim=64, max_position_embeddings=128)
        cos, sin = rotary(seq_len=32)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    def test_bfloat16(self):
        rotary = RotaryEmbedding(head_dim=64, max_position_embeddings=128)
        q = torch.randn(2, 4, 16, 64, dtype=torch.bfloat16)
        k = torch.randn(2, 4, 16, 64, dtype=torch.bfloat16)
        cos, sin = rotary(seq_len=16, dtype=torch.bfloat16)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.dtype == torch.bfloat16
        assert k_rot.dtype == torch.bfloat16

    def test_explicit_position_ids(self, rotary: RotaryEmbedding):
        position_ids = torch.tensor([3, 7, 11])
        cos, _sin = rotary(seq_len=3, position_ids=position_ids)
        assert cos.shape == (3, 32)

    def test_invalid_odd_head_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            RotaryEmbedding(head_dim=63)


class TestApplyRotaryPosEmb:
    def test_shape_preserved(self):
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        cos = torch.randn(10, 32)
        sin = torch.randn(10, 32)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_broadcast_with_batch_dim(self):
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        cos = torch.randn(1, 1, 10, 32)
        sin = torch.randn(1, 1, 10, 32)
        q_out, _k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape

    def test_norm_preserved(self):
        q = torch.randn(2, 4, 8, 32)
        theta = torch.randn(8, 16)
        cos = theta.cos()
        sin = theta.sin()
        norm_before = q.norm(dim=-1)
        q_out, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
        norm_after = q_out.norm(dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5)
