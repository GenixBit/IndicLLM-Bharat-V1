"""Rotary Positional Embedding (RoPE) with YaRN and Dynamic-NTK Long-Context Scaling.

Supports standard RoPE, Linear Interpolation, Dynamic NTK-aware scaling, and YaRN
(Yet another RoPE extensioN) to scale context windows up to 32k - 128k tokens for Bharat-1B.
"""

from __future__ import annotations

import math
import typing
from typing import Any

import torch
import torch.nn as nn


def _compute_yarn_inv_freq(
    head_dim: int,
    rope_theta: float = 10000.0,
    factor: float = 8.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """Compute YaRN (Yet another RoPE extensioN) interpolated inverse frequencies."""
    pos = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (pos / head_dim))

    # Wavelength calculation for each dimension pair: lambda = 2 * pi / freq
    wavelengths = 2.0 * math.pi / inv_freq
    l_0 = float(original_max_position_embeddings)

    # Ramp function: low freq (lambda > l_0 / beta_slow) -> interpolate (factor)
    # high freq (lambda < l_0 / beta_fast) -> do not interpolate
    # middle freq -> linear ramp between 0 and 1
    low = l_0 / beta_fast
    high = l_0 / beta_slow

    # gamma = 0 -> low freq (pure interpolation), gamma = 1 -> high freq (no interpolation)
    gamma = (wavelengths - low) / max(1e-5, (high - low))
    gamma = torch.clamp(gamma, 0.0, 1.0)

    # Interpolated frequencies: (1 - gamma) * (inv_freq / factor) + gamma * inv_freq
    yarn_inv_freq = (1.0 - gamma) * (inv_freq / factor) + gamma * inv_freq
    return yarn_inv_freq


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding with YaRN, Dynamic-NTK, and Linear long-context scaling.

    Supports:
        - Baseline RoPE (no scaling)
        - 'yarn': YaRN ramp interpolation with attention temperature scaling
        - 'dynamic_ntk': Dynamic NTK-aware frequency scaling for sequence extension
        - 'linear': Linear positional frequency interpolation

    All frequency calculations and trigonometric functions are computed in float32.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10_000.0,
        rope_scaling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
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
        self.rope_scaling = rope_scaling

        self.scaling_type: str | None = None
        self.scaling_factor: float = 1.0
        self.original_max_pos: int = max_position_embeddings
        self.yarn_temp: float = 1.0

        if rope_scaling is not None:
            self.scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            self.scaling_factor = float(rope_scaling.get("factor", 1.0))
            self.original_max_pos = int(
                rope_scaling.get("original_max_position_embeddings", max_position_embeddings)
            )

        # Compute initial base inverse frequencies
        if self.scaling_type == "yarn":
            inv_freq = _compute_yarn_inv_freq(
                head_dim=head_dim,
                rope_theta=rope_theta,
                factor=self.scaling_factor,
                original_max_position_embeddings=self.original_max_pos,
            )
            # Temperature scaling for YaRN: t = 0.1 * ln(s) + 1.0
            self.yarn_temp = 0.1 * math.log(max(1.0, self.scaling_factor)) + 1.0
        elif self.scaling_type == "linear":
            pos = torch.arange(0, head_dim, 2, dtype=torch.float32)
            inv_freq = 1.0 / (rope_theta ** (pos / head_dim)) / self.scaling_factor
        else:
            # Default standard RoPE & dynamic_ntk initial base
            pos = torch.arange(0, head_dim, 2, dtype=torch.float32)
            inv_freq = 1.0 / (rope_theta ** (pos / head_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def _get_dynamic_ntk_inv_freq(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute dynamic NTK-aware inverse frequencies on the fly when seq_len exceeds base."""
        if seq_len <= self.original_max_pos:
            return typing.cast(torch.Tensor, self.inv_freq.to(device=device))

        factor = float(seq_len) / float(self.original_max_pos)
        dim = self.head_dim
        # Base theta scaled by factor ** (dim / (dim - 2))
        scaled_theta = self.rope_theta * (factor ** (dim / (dim - 2)))
        pos = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        return typing.cast(torch.Tensor, 1.0 / (scaled_theta ** (pos / dim)))

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

        buf_device: torch.device = typing.cast(torch.device, self.inv_freq.device)

        if position_ids is not None:
            if position_ids.dim() not in (1, 2):
                raise ValueError(f"position_ids must be 1-D or 2-D, got {position_ids.dim()}-D")
            expected_seq = seq_len
            if position_ids.dim() == 1:
                if position_ids.shape[0] != expected_seq:
                    raise ValueError(
                        f"position_ids length ({position_ids.shape[0]}) must match "
                        f"seq_len ({expected_seq})"
                    )
            else:
                if position_ids.shape[1] != expected_seq:
                    raise ValueError(
                        f"position_ids sequence length ({position_ids.shape[1]}) must "
                        f"match seq_len ({expected_seq})"
                    )
            if position_ids.numel() > 0 and position_ids.min() < 0:
                raise ValueError(
                    f"position_ids must not contain negative values, "
                    f"got min={position_ids.min().item()}"
                )
            position_ids = position_ids.to(device=buf_device)
        else:
            position_ids = torch.arange(
                offset,
                offset + seq_len,
                dtype=torch.float32,
                device=buf_device,
            )

        # Dynamic NTK frequency calculation if requested
        if self.scaling_type == "dynamic_ntk":
            inv_freq_f32 = self._get_dynamic_ntk_inv_freq(seq_len, buf_device)
        else:
            inv_freq_f32 = self.inv_freq.float()

        positions_f32 = position_ids.float()

        if position_ids.dim() == 1:
            freqs = torch.einsum("i,j->ij", positions_f32, inv_freq_f32)
        elif position_ids.dim() == 2:
            freqs = torch.einsum("bi,j->bij", positions_f32, inv_freq_f32)
        else:
            raise ValueError(f"position_ids must be 1-D or 2-D, got {position_ids.dim()}-D")

        cos = freqs.cos()
        sin = freqs.sin()

        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)

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
        Tuple of rotated ``(q, k)`` with the same shapes and dtype as inputs.
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
    dtype = x.dtype
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    cos_dim = cos.dim()
    if cos_dim == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos_dim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Cast cos/sin to input dtype (they were computed in float32)
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)

    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)
