from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Architecture:
        gate = SiLU(gate_proj(x))
        up = up_proj(x)
        output = down_proj(gate * up)

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
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))  # type: ignore[no-any-return]
