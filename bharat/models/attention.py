from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bharat.models.config import BharatModelConfig
from bharat.models.rotary import RotaryEmbedding, apply_rotary_pos_emb


def _build_combined_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """
    Build a combined causal + padding attention mask.

    ``attention_mask`` (optional) shape ``(batch_size, sequence_length)``:
        - ``1`` or ``True`` = valid (keep) token
        - ``0`` or ``False`` = padding (mask out) token

    Returns ``None`` when no padding mask is supplied (caller should fall back to
    ``is_causal=True``), or a 4-D float mask broadcastable to
    ``(batch_size, num_heads, query_length, key_length)``.
    """
    if attention_mask is None:
        return None

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2-D (batch_size, sequence_length), "
            f"got {attention_mask.dim()}-D"
        )

    _batch_size, mask_seq_len = attention_mask.shape
    if mask_seq_len != key_length:
        raise ValueError(
            f"attention_mask sequence length ({mask_seq_len}) must match key_length ({key_length})"
        )

    # Normalise to float: 0.0 = keep, -inf = mask
    pad_mask = attention_mask.to(dtype=dtype, device=device)
    pad_mask = torch.where(pad_mask > 0.5, 0.0, float("-inf"))
    # (batch, 1, 1, key)
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

    # Causal mask: (1, 1, query, key), lower-triangular
    # Start with all -inf, then zero out positions where key <= query
    causal_mask = torch.full(
        (1, 1, query_length, key_length), float("-inf"), dtype=dtype, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    # Combine: both use -inf semantics, so element-wise max (or addition) works
    return causal_mask + pad_mask


class GroupedQueryAttention(nn.Module):
    """
    Grouped-query causal attention with Rotary position embeddings.

    Uses separate Q, K, V projections (one each per head group).
    Q has ``num_attention_heads`` heads, K and V have ``num_key_value_heads``.
    K/V heads are repeated via ``repeat_interleave`` to match the query-head count.

    Tensor shapes (forward):
        Input:  ``(batch_size, seq_len, hidden_size)``
        Output: ``(batch_size, seq_len, hidden_size)``

    Attention mask semantics:
        - ``attention_mask`` can be ``None`` (no padding mask, causal only).
        - If provided, shape must be ``(batch_size, sequence_length)``
          with ``1`` or ``True`` for valid (keep) and ``0`` or ``False`` for padding (mask).
        - Causal masking **and** padding masking are always applied together
          when ``attention_mask`` is supplied.
    """

    def __init__(self, config: BharatModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos: torch.Tensor
        sin: torch.Tensor
        cos, sin = self.rotary(
            seq_len=seq_len,
            position_ids=position_ids,
            dtype=hidden_states.dtype,
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        combined_mask = _build_combined_mask(
            attention_mask=attention_mask,
            query_length=seq_len,
            key_length=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output: torch.Tensor = self.o_proj(attn_output)
        return output
