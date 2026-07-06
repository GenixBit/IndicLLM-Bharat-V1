from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BharatModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.vocab_size <= 0:
            errors.append(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {self.hidden_size}")
        if self.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive, got {self.intermediate_size}")
        if self.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            errors.append(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.num_key_value_heads <= 0:
            errors.append(f"num_key_value_heads must be positive, got {self.num_key_value_heads}")
        if self.max_position_embeddings <= 0:
            errors.append(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )

        if self.hidden_size % self.num_attention_heads != 0:
            errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            errors.append(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            errors.append(
                f"num_key_value_heads ({self.num_key_value_heads}) must not exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        head_dim = self.head_dim
        if head_dim % 2 != 0:
            errors.append(
                f"head_dim ({head_dim}) must be even for rotary embeddings. "
                f"hidden_size ({self.hidden_size}) / num_attention_heads "
                f"({self.num_attention_heads}) = {head_dim}"
            )

        if not 0.0 <= self.attention_dropout <= 1.0:
            errors.append(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")
        if not 0.0 <= self.hidden_dropout <= 1.0:
            errors.append(f"hidden_dropout must be in [0, 1], got {self.hidden_dropout}")
        if self.rope_theta <= 0:
            errors.append(f"rope_theta must be positive, got {self.rope_theta}")
        if self.rms_norm_eps <= 0:
            errors.append(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")

        if errors:
            raise ValueError("BharatModelConfig validation failed:\n" + "\n".join(errors))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BharatModelConfig:
        return cls(**d)
