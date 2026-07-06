from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Architecture:
        gate = SiLU(gate_proj(x))
        up = up_proj(x)
        hidden = gate * up
        output = down_proj(hidden)

    Dropout (if any) is applied to the final output.

    Shapes:
        Input:  (batch, seq, hidden_size)
        Output: (batch, seq, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        projected = self.down_proj(hidden)
        dropped: torch.Tensor = self.dropout(projected)
        return dropped
