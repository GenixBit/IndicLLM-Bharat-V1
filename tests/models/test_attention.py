from __future__ import annotations

import pytest
import torch

from bharat.models.attention import GroupedQueryAttention, _build_combined_mask
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


class TestAttentionMask:
    """Tests for ``_build_combined_mask`` and general mask semantics."""

    def test_none_when_no_mask(self):
        mask = _build_combined_mask(None, 4, 4, "cpu", torch.float32)
        assert mask is None

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _build_combined_mask(torch.randn(2, 1, 8), 4, 4, "cpu", torch.float32)

    def test_rejects_wrong_seq_len(self):
        with pytest.raises(ValueError, match="sequence length"):
            _build_combined_mask(torch.ones(2, 8, dtype=torch.long), 4, 16, "cpu", torch.float32)

    def test_causal_and_padding_combined(self):
        batch, seq = 2, 4
        mask = _build_combined_mask(
            torch.ones(batch, seq, dtype=torch.long), seq, seq, "cpu", torch.float32
        )
        assert mask is not None
        # Combined shape: (batch, 1, query, key) where query == key == seq
        assert mask.shape == (batch, 1, seq, seq)
        # All valid — causal mask is the only restriction: lower triangular
        assert mask[0, 0, 0, 0].item() == 0.0  # position 0 attends to position 0
        assert mask[0, 0, 1, 0].item() == 0.0  # position 1 attends to position 0
        assert mask[0, 0, 0, 1].item() == float("-inf")  # position 0 cannot attend to 1

    def test_padding_masked_token_isolated(self):
        batch, seq = 1, 4
        # Mask out last position
        pad_mask = torch.ones(batch, seq, dtype=torch.long)
        pad_mask[0, -1] = 0
        mask = _build_combined_mask(pad_mask, seq, seq, "cpu", torch.float32)
        # All query positions have -inf for the last key
        for q in range(seq):
            assert mask[0, 0, q, -1].item() == float("-inf")
        # Non-masked positions still causal
        assert mask[0, 0, 0, 0].item() == 0.0


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

        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(256)
        out_modified = attn(x_modified)

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

        x2 = x.clone()
        x2[:, prefix_len:] = torch.randn(batch, seq - prefix_len, hidden)

        out1 = attn(x)
        out2 = attn(x2)

        for pos in range(prefix_len):
            assert torch.allclose(out1[:, pos], out2[:, pos], atol=1e-5), (
                f"Position {pos} differs despite identical prefix"
            )

    # ---------- Causal with mask (future masked tokens cannot affect)
    def test_causal_no_leakage_with_mask(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch = 2
        seq = 8
        hidden = 256
        prefix_len = 4

        x = torch.randn(batch, seq, hidden)
        x2 = x.clone()
        x2[:, prefix_len:] = torch.randn(batch, seq - prefix_len, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)

        out1 = attn(x, attention_mask=mask)
        out2 = attn(x2, attention_mask=mask)

        for pos in range(prefix_len):
            assert torch.allclose(out1[:, pos], out2[:, pos], atol=1e-5), (
                f"Position {pos} differs despite identical prefix (with mask)"
            )

    # ---------- Masked token cannot affect valid outputs ----------

    def test_masked_token_isolated(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 1, 4, 256
        x = torch.randn(batch, seq, hidden)

        # Mask out position 2
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, 2] = 0

        out = attn(x, attention_mask=mask)

        # Change the masked position
        x2 = x.clone()
        x2[0, 2] = torch.randn(hidden)
        out2 = attn(x2, attention_mask=mask)

        # Valid positions (0, 1, 3) should be unchanged
        for pos in [0, 1, 3]:
            assert torch.allclose(out[0, pos], out2[0, pos], atol=1e-5), (
                f"Position {pos} changed despite masked key at position 2"
            )

    # ---------- Unmasked past token affects later outputs ----------

    def test_unmasked_past_affects_future(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 1, 4, 256
        x = torch.randn(batch, seq, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)
        out = attn(x, attention_mask=mask)

        x2 = x.clone()
        x2[0, 0] = torch.randn(hidden)
        out2 = attn(x2, attention_mask=mask)

        # Position 1 should be different since attention[1] depends on position 0
        assert not torch.allclose(out[0, 1], out2[0, 1], atol=1e-5), (
            "Changing position 0 should affect position 1 output"
        )

    # ---------- Batch-specific padding masks ----------

    def test_batch_specific_padding(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 2, 4, 256
        x = torch.randn(batch, seq, hidden)

        # First sample: mask out last token; second: all valid
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, -1] = 0

        out = attn(x, attention_mask=mask)

        # Changing masked token in sample 0 should not affect sample 0
        x2 = x.clone()
        x2[0, -1] = torch.randn(hidden)
        out2 = attn(x2, attention_mask=mask)

        # Sample 0, position 0 (valid) should be unchanged
        assert torch.allclose(out[0, 0], out2[0, 0], atol=1e-5), (
            "Masked token change in sample 0 affected sample 0, position 0"
        )

    # ---------- Malformed mask shapes ----------

    def test_malformed_mask_raises(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(1, 4, 256)

        # 3-D mask
        with pytest.raises(ValueError, match=r"must be 2-D|attention_mask"):
            attn(x, attention_mask=torch.randn(1, 1, 4))

        # Wrong sequence length
        with pytest.raises(ValueError, match="sequence length"):
            attn(x, attention_mask=torch.ones(1, 8, dtype=torch.long))

    # ---------- MHA, GQA, MQA all preserve causality with masks ----------

    @pytest.mark.parametrize("kv_heads", [8, 4, 1])
    def test_all_modes_preserve_causality_with_mask(self, kv_heads: int):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=kv_heads, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 2, 6, 256
        prefix_len = 3

        x = torch.randn(batch, seq, hidden)
        x2 = x.clone()
        x2[:, prefix_len:] = torch.randn(batch, seq - prefix_len, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)
        out1 = attn(x, attention_mask=mask)
        out2 = attn(x2, attention_mask=mask)

        for pos in range(prefix_len):
            assert torch.allclose(out1[:, pos], out2[:, pos], atol=1e-5), (
                f"MQA/GQA/MHA mode {kv_heads} position {pos} differs with mask"
            )

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

    # ---------- Reference attention test ----------

    def test_reference_attention(self):
        cfg = _make_config(
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=16,
            max_position_embeddings=16,
        )
        torch.manual_seed(42)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        # Override rotary to identity so we can compare against manual matmul
        # Use a fixed cos (all 1) and sin (all 0) → no rotation
        batch, seq, hidden = 1, 3, 16
        num_heads = 2
        head_dim = hidden // num_heads  # 8
        half_dim = head_dim // 2  # 4

        cos = torch.ones(seq, half_dim)
        sin = torch.zeros(seq, half_dim)

        class IdentityRoPE(torch.nn.Module):
            def forward(self, seq_len, position_ids=None, offset=0, device=None, dtype=None):
                return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

        attn.rotary = IdentityRoPE()

        x = torch.randn(batch, seq, hidden)

        # Run our implementation without dropout
        out = attn(x)

        # Manual reference: Q = x @ W_q, K = x @ W_k, V = x @ W_v
        w_q = attn.q_proj.weight.T  # (hidden, num_heads*head_dim)
        w_k = attn.k_proj.weight.T
        w_v = attn.v_proj.weight.T
        w_o = attn.o_proj.weight.T

        q_ref = x @ w_q  # (1, 3, 16)
        k_ref = x @ w_k
        v_ref = x @ w_v

        q_ref = q_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k_ref = k_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        v_ref = v_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)

        # Scaled dot product (manual)
        scale = head_dim**0.5
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) / scale

        # Causal mask
        causal = torch.full((seq, seq), float("-inf"))
        causal = torch.triu(causal, diagonal=1)  # upper triangle → mask out
        scores = scores + causal

        attn_weights = torch.softmax(scores, dim=-1)
        attn_ref = torch.matmul(attn_weights, v_ref)
        attn_ref = attn_ref.transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_ref = attn_ref @ w_o

        assert torch.allclose(out, attn_ref, atol=1e-5), (
            "Reference attention computation does not match implementation"
        )

    # ---------- Padding mask forward finite ----------

    def test_padding_mask_forward(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 2, 8, 256
        x = torch.randn(batch, seq, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, -2:] = 0

        out = attn(x, attention_mask=mask)
        assert out.shape == (batch, seq, hidden)
        assert torch.isfinite(out).all(), "NaNs produced with padding mask"

    # ---------- All-valid mask matches no-mask ----------

    def test_all_valid_mask_matches_no_mask(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        x = torch.randn(2, 8, 256)
        mask = torch.ones(2, 8, dtype=torch.long)

        out_no_mask = attn(x)
        out_with_mask = attn(x, attention_mask=mask)

        assert torch.allclose(out_no_mask, out_with_mask, atol=1e-5), (
            "All-valid mask produces different output from no mask"
        )
