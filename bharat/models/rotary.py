from __future__ import annotations

import typing

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding using interleaved even/odd rotation convention.

    For each pair of dimensions (2k, 2k+1) at position p:
        x_2k   -> x_2k * cos(p * theta_k) - x_{2k+1} * sin(p * theta_k)
        x_{2k+1} -> x_2k * sin(p * theta_k) + x_{2k+1} * cos(p * theta_k)

    where theta_k = 1 / (rope_theta ** (2k / head_dim)).

    The returned ``cos`` / ``sin`` tensors have shape:
        - ``(sequence_length, head_dim // 2)`` when ``position_ids`` is 1-D or ``None``.
        - ``(batch_size, 1, sequence_length, head_dim // 2)`` when
          ``position_ids`` is 2-D ``(batch_size, sequence_length)``.

    .. note::
        Frequencies beyond ``max_position_embeddings`` are computed on the fly;
        no NTK-aware / YaRN / LongRoPE scaling is applied.
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
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
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
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")

        compute_dtype = dtype if dtype else torch.float32

        buf_device: torch.device = typing.cast(torch.device, self.inv_freq.device)

        if position_ids is not None:
            position_ids = position_ids.to(device=buf_device)
        else:
            position_ids = torch.arange(
                offset,
                offset + seq_len,
                dtype=torch.float32,
                device=buf_device,
            )

        inv_freq = self.inv_freq.to(dtype=dtype) if dtype else self.inv_freq

        if position_ids.dim() == 1:
            freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq.float())
        elif position_ids.dim() == 2:
            freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq.float())
        else:
            raise ValueError(f"position_ids must be 1-D or 2-D, got {position_ids.dim()}-D")

        cos = freqs.cos().to(dtype=compute_dtype)
        sin = freqs.sin().to(dtype=compute_dtype)
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
        q: Query tensor of shape ``(batch, heads, seq, head_dim)``.
        k: Key tensor of shape ``(batch, heads, seq, head_dim)``.
        cos: Cosine values broadcastable to query/key heads.
        sin: Sine values broadcastable to query/key heads.

    Returns:
        Tuple of rotated ``(q, k)`` with the same shapes as inputs.
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

    cos_dim = cos.dim()
    if cos_dim == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos_dim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
