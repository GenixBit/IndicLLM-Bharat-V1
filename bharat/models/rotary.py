from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding using interleaved even/odd rotation convention.

    For each pair of dimensions (2k, 2k+1) at position p:
        x_2k   -> x_2k * cos(p * theta_k) - x_{2k+1} * sin(p * theta_k)
        x_{2k+1} -> x_2k * sin(p * theta_k) + x_{2k+1} * cos(p * theta_k)

    where theta_k = 1 / (rope_theta ** (2k / head_dim)).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for rotary embeddings, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self,
        seq_len: int,
        position_ids: torch.Tensor | None = None,
        offset: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            position_ids = position_ids.float()
        else:
            total_len = offset + seq_len
            if total_len > self.max_position_embeddings:
                total_len = max(total_len, self.max_position_embeddings)
            position_ids = torch.arange(
                offset,
                total_len,
                dtype=torch.float32,
                device=self.inv_freq.device,  # type: ignore[arg-type]
            )

        inv_freq = (
            self.inv_freq.to(dtype=dtype, device=device) if dtype or device else self.inv_freq
        )
        freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings using interleaved even/odd convention.

    Args:
        q: Query tensor of shape (batch, heads, seq, head_dim).
        k: Key tensor of shape (batch, heads, seq, head_dim).
        cos: Cosine values of shape (seq, head_dim) or (batch, 1, seq, head_dim).
        sin: Sine values of shape (seq, head_dim) or (batch, 1, seq, head_dim).

    Returns:
        Tuple of rotated (q, k) with same shapes as inputs.
    """
    q_embed = _rotate_half(q, cos, sin)
    k_embed = _rotate_half(k, cos, sin)
    return q_embed, k_embed


def _rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved rotation to the last dimension of x."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
