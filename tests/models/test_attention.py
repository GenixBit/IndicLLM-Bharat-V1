from __future__ import annotations

import pytest
import torch

from bharat.models.attention import GroupedQueryAttention
from bharat.models.config import BharatModelConfig


def _make_config(
    num_attention_heads: int = 8,
    num_key_value_heads: int = 8,
    hidden_size: int = 512,
    intermediate_size: int = 2048,
    num_hidden_layers: int = 2,
    max_position_embeddings: int = 128,
) -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


class TestGroupedQueryAttention:
    # ---------- MHA mode (num_heads == num_kv_heads) ----------

    def test_mha_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out = attn(x)
        assert out.shape == (2, 16, 512)

    def test_mha_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_mha_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- GQA mode (num_heads > num_kv_heads) ----------

    def test_gqa_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out = attn(x)
        assert out.shape == (2, 16, 512)

    def test_gqa_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_gqa_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- MQA mode (num_kv_heads == 1) ----------

    def test_mqa_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out = attn(x)
        assert out.shape == (2, 16, 512)

    def test_mqa_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out = attn(x)
        assert torch.isfinite(out).all()

    def test_mqa_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- Gradient checks on all projections ----------

    def test_gradients_on_projections(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.o_proj.weight.grad is not None
        assert torch.isfinite(attn.q_proj.weight.grad).all()
        assert torch.isfinite(attn.k_proj.weight.grad).all()
        assert torch.isfinite(attn.v_proj.weight.grad).all()
        assert torch.isfinite(attn.o_proj.weight.grad).all()

    # ---------- Causal behavior ----------

    def test_causal_behavior(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        x = torch.randn(1, 8, 256)
        out = attn(x)

        # Position i should not attend to j > i.  Verify by checking that
        # changing the last token does not affect the first token's output.
        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(256)
        out_modified = attn(x_modified)

        # First token output should be identical (within tolerance)
        assert torch.allclose(out[0, 0], out_modified[0, 0], atol=1e-5)

    # ---------- Causal leakage test (explicit) ----------

    def test_causal_no_leakage(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch = 2
        seq = 8
        hidden = 256
        prefix_len = 4

        x = torch.randn(batch, seq, hidden)

        # Create second sequence with identical prefix but different suffix
        x2 = x.clone()
        x2[:, prefix_len:] = torch.randn(batch, seq - prefix_len, hidden)

        out1 = attn(x)
        out2 = attn(x2)

        # Positions before the divergence point must be equal
        for pos in range(prefix_len):
            assert torch.allclose(out1[:, pos], out2[:, pos], atol=1e-5), (
                f"Position {pos} differs despite identical prefix"
            )

    # ---------- Padding mask ----------

    def test_padding_mask_behavior(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 2, 8, 256
        x = torch.randn(batch, seq, hidden)

        # Mask out the last token in the first sequence
        mask = torch.zeros(batch, 1, 1, seq)
        mask[0, 0, 0, -1] = float("-inf")

        out_no_mask = attn(x)
        out_with_mask = attn(x, attention_mask=mask)

        # Output shapes match
        assert out_no_mask.shape == out_with_mask.shape

    # ---------- Sequence length 1 ----------

    def test_sequence_length_one(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 1, 256)
        out = attn(x)
        assert out.shape == (2, 1, 256)
        assert torch.isfinite(out).all()

    # ---------- Variable batch sizes ----------

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(self, batch_size: int):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(batch_size, 8, 256)
        out = attn(x)
        assert out.shape == (batch_size, 8, 256)

    # ---------- Float 32 ----------

    def test_float32(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 256, dtype=torch.float32)
        out = attn(x)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    # ---------- BFloat16 ----------

    def test_bfloat16(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg).to(dtype=torch.bfloat16)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16)
        out = attn(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    # ---------- State dict round trip ----------

    def test_state_dict_roundtrip(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        state = attn.state_dict()
        loaded = GroupedQueryAttention(cfg)
        loaded.load_state_dict(state)
        x = torch.randn(2, 8, 256)
        assert torch.allclose(attn(x), loaded(x), atol=1e-5)

    # ---------- Deterministic in eval mode ----------

    def test_deterministic_eval(self):
        cfg = _make_config(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=256,
            max_position_embeddings=128,
        )
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x = torch.randn(2, 8, 256)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2, atol=1e-5)
