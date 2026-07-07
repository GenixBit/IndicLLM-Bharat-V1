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
        mask = _build_combined_mask(None, 4, 4, 2, "cpu", torch.float32)
        assert mask is None

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            _build_combined_mask(torch.randn(2, 1, 8), 4, 4, 2, "cpu", torch.float32)

    def test_rejects_wrong_seq_len(self):
        with pytest.raises(ValueError, match="sequence length"):
            _build_combined_mask(torch.ones(2, 8, dtype=torch.long), 4, 16, 2, "cpu", torch.float32)

    def test_rejects_wrong_batch_size(self):
        with pytest.raises(ValueError, match="batch size"):
            _build_combined_mask(torch.ones(3, 4, dtype=torch.long), 4, 4, 2, "cpu", torch.float32)

    def test_causal_and_padding_combined(self):
        batch, seq = 2, 4
        mask = _build_combined_mask(
            torch.ones(batch, seq, dtype=torch.long), seq, seq, batch, "cpu", torch.float32
        )
        assert mask is not None
        assert mask.shape == (batch, 1, seq, seq)
        assert mask[0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 1, 0].item() == 0.0
        assert mask[0, 0, 0, 1].item() == float("-inf")

    def test_padding_masked_token_isolated(self):
        batch, seq = 1, 4
        pad_mask = torch.ones(batch, seq, dtype=torch.long)
        pad_mask[0, -1] = 0
        mask = _build_combined_mask(pad_mask, seq, seq, batch, "cpu", torch.float32)
        for q in range(seq):
            assert mask[0, 0, q, -1].item() == float("-inf")
        assert mask[0, 0, 0, 0].item() == 0.0

    def test_boolean_mask_valid(self):
        batch, seq = 2, 4
        mask = _build_combined_mask(
            torch.ones(batch, seq, dtype=torch.bool), seq, seq, batch, "cpu", torch.float32
        )
        assert mask is not None
        assert mask.shape == (batch, 1, seq, seq)

    def test_integer_mask_valid(self):
        batch, seq = 2, 4
        mask = _build_combined_mask(
            torch.ones(batch, seq, dtype=torch.long), seq, seq, batch, "cpu", torch.float32
        )
        assert mask is not None
        assert mask.shape == (batch, 1, seq, seq)

    def test_right_padded_batch(self):
        batch, seq = 2, 6
        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, -2:] = 0  # sample 0: right-padded
        mask[1, -3:] = 0  # sample 1: right-padded by 3
        result = _build_combined_mask(mask, seq, seq, batch, "cpu", torch.float32)
        assert result is not None
        assert result.shape == (batch, 1, seq, seq)
        # Check both samples have correct -inf at padded positions
        assert result[0, 0, 0, -1].item() == float("-inf")
        assert result[0, 0, 0, -2].item() == float("-inf")
        assert result[1, 0, 0, -1].item() == float("-inf")
        assert result[1, 0, 0, -2].item() == float("-inf")
        assert result[1, 0, 0, -3].item() == float("-inf")


class TestGroupedQueryAttention:
    # ---------- MHA mode (num_heads == num_kv_heads) ----------

    def test_mha_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out, _cache = attn(x)
        assert out.shape == (2, 16, 512)

    def test_mha_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out, _cache = attn(x)
        assert torch.isfinite(out).all()

    def test_mha_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=8)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out, _cache = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- GQA mode (num_heads > num_kv_heads) ----------

    def test_gqa_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out, _cache = attn(x)
        assert out.shape == (2, 16, 512)

    def test_gqa_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out, _cache = attn(x)
        assert torch.isfinite(out).all()

    def test_gqa_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out, _cache = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- MQA mode (num_kv_heads == 1) ----------

    def test_mqa_output_shape(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 512)
        out, _cache = attn(x)
        assert out.shape == (2, 16, 512)

    def test_mqa_forward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out, _cache = attn(x)
        assert torch.isfinite(out).all()

    def test_mqa_backward(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512, requires_grad=True)
        out, _cache = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    # ---------- Gradient checks on all projections ----------

    def test_gradients_on_projections(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 512)
        out, _cache = attn(x)
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
        out, _cache = attn(x)

        x_modified = x.clone()
        x_modified[0, -1] = torch.randn(256)
        out_modified, _cache = attn(x_modified)

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

        out1, _cache = attn(x)
        out2, _cache = attn(x2)

        for pos in range(prefix_len):
            assert torch.allclose(
                out1[:, pos], out2[:, pos], atol=1e-5
            ), f"Position {pos} differs despite identical prefix"

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

        out1, _cache = attn(x, attention_mask=mask)
        out2, _cache = attn(x2, attention_mask=mask)

        for pos in range(prefix_len):
            assert torch.allclose(
                out1[:, pos], out2[:, pos], atol=1e-5
            ), f"Position {pos} differs despite identical prefix (with mask)"

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

        out, _cache = attn(x, attention_mask=mask)

        # Change the masked position
        x2 = x.clone()
        x2[0, 2] = torch.randn(hidden)
        out2, _cache = attn(x2, attention_mask=mask)

        # Valid positions (0, 1, 3) should be unchanged
        for pos in [0, 1, 3]:
            assert torch.allclose(
                out[0, pos], out2[0, pos], atol=1e-5
            ), f"Position {pos} changed despite masked key at position 2"

    # ---------- Unmasked past token affects later outputs ----------

    def test_unmasked_past_affects_future(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 1, 4, 256
        x = torch.randn(batch, seq, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)
        out, _cache = attn(x, attention_mask=mask)

        x2 = x.clone()
        x2[0, 0] = torch.randn(hidden)
        out2, _cache = attn(x2, attention_mask=mask)

        # Position 1 should be different since attention[1] depends on position 0
        assert not torch.allclose(
            out[0, 1], out2[0, 1], atol=1e-5
        ), "Changing position 0 should affect position 1 output"

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

        out, _cache = attn(x, attention_mask=mask)

        # Changing masked token in sample 0 should not affect sample 0
        x2 = x.clone()
        x2[0, -1] = torch.randn(hidden)
        out2, _cache = attn(x2, attention_mask=mask)

        # Sample 0, position 0 (valid) should be unchanged
        assert torch.allclose(
            out[0, 0], out2[0, 0], atol=1e-5
        ), "Masked token change in sample 0 affected sample 0, position 0"

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

        # Wrong batch size
        with pytest.raises(ValueError, match="batch size"):
            attn(x, attention_mask=torch.ones(2, 4, dtype=torch.long))

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
        out1, _cache = attn(x, attention_mask=mask)
        out2, _cache = attn(x2, attention_mask=mask)

        for pos in range(prefix_len):
            assert torch.allclose(
                out1[:, pos], out2[:, pos], atol=1e-5
            ), f"MQA/GQA/MHA mode {kv_heads} position {pos} differs with mask"

    # ---------- Sequence length 1 ----------

    def test_sequence_length_one(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 1, 256)
        out, _cache = attn(x)
        assert out.shape == (2, 1, 256)
        assert torch.isfinite(out).all()

    # ---------- Variable batch sizes ----------

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(self, batch_size: int):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(batch_size, 8, 256)
        out, _cache = attn(x)
        assert out.shape == (batch_size, 8, 256)

    # ---------- Float 32 ----------

    def test_float32(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 8, 256, dtype=torch.float32)
        out, _cache = attn(x)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    # ---------- BFloat16 ----------

    def test_bfloat16(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg).to(dtype=torch.bfloat16)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16)
        out, _cache = attn(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_bfloat16_rotary_precision(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg).to(dtype=torch.bfloat16)
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16)
        out, _cache = attn(x)
        assert torch.isfinite(out.float()).all()

    # ---------- State dict round trip ----------

    def test_state_dict_roundtrip(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        state = attn.state_dict()
        loaded = GroupedQueryAttention(cfg)
        loaded.load_state_dict(state)
        x = torch.randn(2, 8, 256)
        out1, _ = attn(x)
        out2, _ = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-5)

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
        out1, _ = attn(x)
        out2, _ = attn(x)
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
        batch, seq, hidden = 1, 3, 16
        num_heads = 2
        head_dim = hidden // num_heads
        half_dim = head_dim // 2

        cos = torch.ones(seq, half_dim)
        sin = torch.zeros(seq, half_dim)

        class IdentityRoPE(torch.nn.Module):
            def forward(self, seq_len, position_ids=None, offset=0, dtype=None):
                return cos.to(dtype=dtype), sin.to(dtype=dtype)

        attn.rotary = IdentityRoPE()

        x = torch.randn(batch, seq, hidden)

        out, _cache = attn(x)

        # Manual reference: Q = x @ W_q, K = x @ W_k, V = x @ W_v
        w_q = attn.q_proj.weight.T
        w_k = attn.k_proj.weight.T
        w_v = attn.v_proj.weight.T
        w_o = attn.o_proj.weight.T

        q_ref = x @ w_q
        k_ref = x @ w_k
        v_ref = x @ w_v

        q_ref = q_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k_ref = k_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        v_ref = v_ref.view(batch, seq, num_heads, head_dim).transpose(1, 2)

        scale = head_dim**0.5
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) / scale

        causal = torch.full((seq, seq), float("-inf"))
        causal = torch.triu(causal, diagonal=1)
        scores = scores + causal

        attn_weights = torch.softmax(scores, dim=-1)
        attn_ref = torch.matmul(attn_weights, v_ref)
        attn_ref = attn_ref.transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_ref = attn_ref @ w_o

        assert torch.allclose(
            out, attn_ref, atol=1e-5
        ), "Reference attention computation does not match implementation"

    # ---------- Padding mask forward finite ----------

    def test_padding_mask_forward(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        batch, seq, hidden = 2, 8, 256
        x = torch.randn(batch, seq, hidden)

        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, -2:] = 0

        out, _cache = attn(x, attention_mask=mask)
        assert out.shape == (batch, seq, hidden)
        assert torch.isfinite(out).all(), "NaNs produced with padding mask"

    # ---------- All-valid mask matches no-mask ----------

    def test_all_valid_mask_matches_no_mask(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()

        x = torch.randn(2, 8, 256)
        mask = torch.ones(2, 8, dtype=torch.long)

        out_no_mask, _ = attn(x)
        out_with_mask, _ = attn(x, attention_mask=mask)

        assert torch.allclose(
            out_no_mask, out_with_mask, atol=1e-5
        ), "All-valid mask produces different output from no mask"

    # ---------- KV cache tests ----------

    def test_cache_not_returned_when_use_cache_false(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x = torch.randn(2, 4, 256)
        _, cache = attn(x, use_cache=False)
        assert cache is None

    def test_cache_returned_when_use_cache_true(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x = torch.randn(2, 4, 256)
        _, cache = attn(x, use_cache=True)
        assert cache is not None
        k, v = cache
        assert k.shape == (2, 4, 4, 64)
        assert v.shape == (2, 4, 4, 64)

    def test_cache_stores_kv_heads_not_query_heads(self):
        cfg = _make_config(num_attention_heads=8, num_key_value_heads=2, hidden_size=512)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x = torch.randn(2, 4, 512)
        _, cache = attn(x, use_cache=True)
        k, v = cache
        assert k.shape[1] == 2
        assert v.shape[1] == 2

    def test_cache_grows(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x1 = torch.randn(2, 4, 256)
        _, cache = attn(x1, use_cache=True)
        assert cache is not None
        assert cache[0].shape[-2] == 4

        x2 = torch.randn(2, 1, 256)
        _, cache = attn(x2, past_key_value=cache, use_cache=True)
        assert cache is not None
        assert cache[0].shape[-2] == 5

    def test_immutable_cache(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x1 = torch.randn(2, 4, 256)
        _, cache = attn(x1, use_cache=True)
        orig_k = cache[0].clone()

        x2 = torch.randn(2, 1, 256)
        _, _ = attn(x2, past_key_value=cache, use_cache=True)

        assert torch.equal(orig_k, cache[0]), "Cache was mutated in place"

    def test_cached_causal_no_leakage(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, seq = 2, 5

        x = torch.randn(batch, seq, 256)
        out_full, _ = attn(x, use_cache=False)

        # Process token by token
        past = None
        out_tokens = []
        for pos in range(seq):
            token_input = x[:, pos : pos + 1, :]
            token_out, past = attn(token_input, past_key_value=past, use_cache=True)
            out_tokens.append(token_out)

        out_cached = torch.cat(out_tokens, dim=1)
        assert torch.allclose(out_full, out_cached, atol=1e-4), "Cached vs full mismatch"

    def test_cached_causal_cannot_see_future(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, seq = 1, 5
        x = torch.randn(batch, seq, 256)

        out_full, _ = attn(x, use_cache=False)

        past = None
        for pos in range(seq):
            token_input = x[:, pos : pos + 1, :]
            token_out, past = attn(token_input, past_key_value=past, use_cache=True)

            expected = out_full[:, pos : pos + 1, :]
            assert torch.allclose(
                token_out, expected, atol=1e-4
            ), f"Position {pos} mismatch in cached decoding"

    def test_cache_with_padding_mask(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4, hidden_size=256)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, seq = 1, 5

        mask = torch.ones(batch, seq, dtype=torch.long)
        mask[0, -2:] = 0  # right-padded

        x = torch.randn(batch, seq, 256)
        out_full, _ = attn(x, attention_mask=mask, use_cache=False)

        past = None
        out_tokens = []
        for pos in range(seq):
            token_input = x[:, pos : pos + 1, :]
            step_mask = mask[:, : pos + 1]
            token_out, past = attn(
                token_input,
                attention_mask=step_mask,
                past_key_value=past,
                use_cache=True,
            )
            out_tokens.append(token_out)

        out_cached = torch.cat(out_tokens, dim=1)
        for pos in range(seq):
            if mask[0, pos] == 1:
                assert torch.allclose(
                    out_full[:, pos], out_cached[:, pos], atol=1e-4
                ), f"Cache with padding mask position {pos} mismatch"

    # ---------- Multi-token cached causality ----------

    def _check_multi_token_causal(self, attn, past, current_tokens, position_ids, hidden_size):
        """Verify that changing a later token in a multi-token cached forward
        does not affect earlier output positions (i.e. causal mask is correct)."""
        seq = current_tokens.shape[1]

        causal_out_a, _ = attn(
            current_tokens,
            position_ids=position_ids,
            past_key_value=past,
            use_cache=True,
        )

        # Change the last token
        current_b = current_tokens.clone()
        current_b[:, -1, :] = torch.randn_like(current_b[:, -1, :])
        pos_ids_b = position_ids.clone()
        pos_ids_b[:, -1] = position_ids[:, -1]

        causal_out_b, _ = attn(
            current_b,
            position_ids=pos_ids_b,
            past_key_value=past,
            use_cache=True,
        )

        # Earlier positions must be identical
        for pos in range(seq - 1):
            assert torch.allclose(
                causal_out_a[:, pos, :], causal_out_b[:, pos, :], atol=1e-4
            ), f"Changing last token altered output at position {pos}"

        # Last position must differ
        assert not torch.allclose(
            causal_out_a[:, -1, :], causal_out_b[:, -1, :], atol=1e-4
        ), "Changing last token should alter its own output"

    def test_cached_multi_token_mha(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, hidden = 1, cfg.hidden_size
        prefix = torch.randn(batch, 3, hidden)
        _, cache = attn(prefix, use_cache=True)

        current = torch.randn(batch, 3, hidden)
        pos_ids = torch.arange(3, 6, dtype=torch.long).unsqueeze(0)
        self._check_multi_token_causal(attn, cache, current, pos_ids, hidden)

    def test_cached_multi_token_gqa(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=2)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, hidden = 1, cfg.hidden_size
        prefix = torch.randn(batch, 3, hidden)
        _, cache = attn(prefix, use_cache=True)

        current = torch.randn(batch, 3, hidden)
        pos_ids = torch.arange(3, 6, dtype=torch.long).unsqueeze(0)
        self._check_multi_token_causal(attn, cache, current, pos_ids, hidden)

    def test_cached_multi_token_mqa(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=1)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, hidden = 1, cfg.hidden_size
        prefix = torch.randn(batch, 3, hidden)
        _, cache = attn(prefix, use_cache=True)

        current = torch.randn(batch, 3, hidden)
        pos_ids = torch.arange(3, 6, dtype=torch.long).unsqueeze(0)
        self._check_multi_token_causal(attn, cache, current, pos_ids, hidden)

    def test_cached_multi_token_batch(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, hidden = 2, cfg.hidden_size
        prefix = torch.randn(batch, 3, hidden)
        _, cache = attn(prefix, use_cache=True)

        current = torch.randn(batch, 3, hidden)
        pos_ids = torch.arange(3, 6, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        self._check_multi_token_causal(attn, cache, current, pos_ids, hidden)

    def test_cached_multi_token_with_padding_mask(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=4)
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        batch, hidden = 1, cfg.hidden_size
        prefix = torch.randn(batch, 3, hidden)
        pad_mask = torch.ones(batch, 3, dtype=torch.long)
        _, cache = attn(prefix, attention_mask=pad_mask, use_cache=True)

        current = torch.randn(batch, 3, hidden)
        full_mask = torch.ones(batch, 6, dtype=torch.long)
        full_mask[0, -2:] = 0
        pos_ids = torch.arange(3, 6, dtype=torch.long).unsqueeze(0)

        causal_out_a, _ = attn(
            current,
            attention_mask=full_mask,
            position_ids=pos_ids,
            past_key_value=cache,
            use_cache=True,
        )

        current_b = current.clone()
        current_b[:, -1, :] = torch.randn_like(current_b[:, -1, :])

        causal_out_b, _ = attn(
            current_b,
            attention_mask=full_mask,
            position_ids=pos_ids,
            past_key_value=cache,
            use_cache=True,
        )

        for pos in range(2):
            assert torch.allclose(
                causal_out_a[:, pos, :], causal_out_b[:, pos, :], atol=1e-4
            ), f"With padding mask, changing last token altered position {pos}"

        assert not torch.allclose(
            causal_out_a[:, -1, :], causal_out_b[:, -1, :], atol=1e-4
        ), "With padding mask, last token output should differ after change"
