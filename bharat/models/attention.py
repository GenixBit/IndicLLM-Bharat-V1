from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bharat.models.config import BharatModelConfig
from bharat.models.rotary import RotaryEmbedding, apply_rotary_pos_emb

# Convenience type alias for a single-layer KV cache
KeyValueCache = tuple[torch.Tensor, torch.Tensor]


def _build_causal_mask(
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
    past_length: int = 0,
) -> torch.Tensor:
    """
    Build an explicit causal mask with offset for KV cache.

    Query index ``q`` may attend to key indices ``<= past_length + q``.
    The causal mask is built via ``torch.triu(..., diagonal=1 + past_length)``
    on an all-``-inf`` tensor.

    Returns a 4-D float mask of shape ``(1, 1, query_length, key_length)``.
    """
    causal_mask = torch.full(
        (1, 1, query_length, key_length), float("-inf"), dtype=dtype, device=device
    )
    causal_mask = torch.triu(causal_mask, diagonal=1 + past_length)
    return causal_mask


def _build_combined_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    past_length: int = 0,
) -> torch.Tensor | None:
    """
    Build a combined causal + padding attention mask.

    ``attention_mask`` (optional) shape ``(batch_size, sequence_length)``:
        - ``1`` or ``True`` = valid (keep) token
        - ``0`` or ``False`` = padding (mask out) token

    The mask is validated to have exactly ``(batch_size, key_length)`` shape.

    ``past_length`` accounts for cached key positions: query index ``q`` may
    attend to key indices ``<= past_length + q``.

    Returns ``None`` only when there is **no** padding mask **and** no offset
    is needed (``past_length == 0`` and ``query_length == 1``). Otherwise
    returns a 4-D float mask broadcastable to
    ``(batch_size, num_heads, query_length, key_length)``.
    """
    if attention_mask is None and past_length == 0:
        return None

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must be 2-D (batch_size, sequence_length), "
                f"got {attention_mask.dim()}-D"
            )
        mask_batch, mask_seq_len = attention_mask.shape
        if mask_batch != batch_size:
            raise ValueError(
                f"attention_mask batch size ({mask_batch}) must match"
                f" input batch size ({batch_size})"
            )
        if mask_seq_len != key_length:
            raise ValueError(
                f"attention_mask sequence length ({mask_seq_len}) must"
                f" match key_length ({key_length})"
            )
        pad_mask = attention_mask.to(dtype=dtype, device=device)
        pad_mask = torch.where(pad_mask > 0.5, 0.0, float("-inf"))
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, key)
    else:
        pad_mask = None

    causal_mask = _build_causal_mask(query_length, key_length, device, dtype, past_length)

    if pad_mask is not None:
        return causal_mask + pad_mask
    return causal_mask


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
        - If provided, shape must be ``(batch_size, key_length)``
          with ``1`` or ``True`` for valid (keep) and ``0`` or ``False`` for padding (mask).
        - Causal masking **and** padding masking are always applied together
          when ``attention_mask`` is supplied.

    KV cache:
        - ``past_key_value`` is a ``(key, value)`` tuple of shape
          ``(batch, num_kv_heads, cached_len, head_dim)``.
        - ``use_cache`` controls whether the updated cache is returned.
        - The cache stores unexpanded K/V (num_kv_heads, not num_heads).
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
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KeyValueCache | None]:
        batch_size, seq_len, _hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Determine past length from cache
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[0].shape[-2]

        # Compute position IDs for current tokens
        if position_ids is None:
            position_ids = (
                torch.arange(
                    past_length,
                    past_length + seq_len,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        cos: torch.Tensor
        sin: torch.Tensor
        cos, sin = self.rotary(
            seq_len=seq_len,
            position_ids=position_ids,
            dtype=hidden_states.dtype,
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate cached K/V (never mutate in-place)
        if past_key_value is not None:
            cached_k, cached_v = past_key_value
            k = torch.cat([cached_k, k], dim=-2)
            v = torch.cat([cached_v, v], dim=-2)

        # Store new cache (unexpanded K/V) before repeating for attention
        new_key_value: KeyValueCache | None = None
        if use_cache:
            new_key_value = (k, v)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        combined_mask = _build_combined_mask(
            attention_mask=attention_mask,
            query_length=seq_len,
            key_length=k.shape[-2],
            batch_size=batch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            past_length=past_length,
        )

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=combined_mask is None,
        )

        output: torch.Tensor = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output, new_key_value
