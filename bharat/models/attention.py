from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bharat.models.config import BharatModelConfig
from bharat.models.rotary import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped-query causal attention with Rotary position embeddings.

    Uses separate Q, K, V projections (one each per head group).
    Q has ``num_attention_heads`` heads, K and V have ``num_key_value_heads``.
    K/V heads are repeated (via expand) to match the query-head count.

    Tensor shapes (forward):
        Input:  (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)

    Attention mask semantics:
        - ``attention_mask`` can be ``None`` (no padding mask, causal only).
        - If provided, shape must be ``(batch_size, 1, 1, seq_len)``
          with 0.0 for keep and -inf for mask (broadcast to heads).
        - Causal masking is handled via ``is_causal=True`` in SDPA.
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

        cos, sin = self.rotary(
            seq_len=seq_len,
            position_ids=position_ids,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)  # type: ignore[no-any-return]
