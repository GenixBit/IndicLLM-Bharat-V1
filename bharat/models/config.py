from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BharatModelConfig:
    """Configuration for Bharat LLM modern decoder architecture."""

    vocab_size: int = 64000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        errors: list[str] = []

        # Check for boolean types passed where ints/floats are expected
        for int_field in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ):
            val = getattr(self, int_field)
            if isinstance(val, bool):
                errors.append(f"{int_field} must be an integer, not bool (got {val})")

        # Integer positive value checks
        if not isinstance(self.vocab_size, bool) and self.vocab_size <= 0:
            errors.append(f"vocab_size must be positive (> 0), got {self.vocab_size}")

        if not isinstance(self.hidden_size, bool) and self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive (> 0), got {self.hidden_size}")

        if not isinstance(self.intermediate_size, bool) and self.intermediate_size <= 0:
            errors.append(f"intermediate_size must be positive (> 0), got {self.intermediate_size}")

        if not isinstance(self.num_hidden_layers, bool) and self.num_hidden_layers <= 0:
            errors.append(f"num_hidden_layers must be positive (> 0), got {self.num_hidden_layers}")

        if not isinstance(self.num_attention_heads, bool) and self.num_attention_heads <= 0:
            errors.append(
                f"num_attention_heads must be positive (> 0), got {self.num_attention_heads}"
            )

        if not isinstance(self.num_key_value_heads, bool) and self.num_key_value_heads <= 0:
            errors.append(
                f"num_key_value_heads must be positive (> 0), got {self.num_key_value_heads}"
            )

        if not isinstance(self.max_position_embeddings, bool) and self.max_position_embeddings <= 0:
            errors.append(
                f"max_position_embeddings must be positive (> 0), got {self.max_position_embeddings}"
            )

        # Relation checks (only if valid positive non-bool numbers)
        if (
            not isinstance(self.num_attention_heads, bool)
            and not isinstance(self.num_key_value_heads, bool)
            and self.num_attention_heads > 0
            and self.num_key_value_heads > 0
        ):
            if self.num_key_value_heads > self.num_attention_heads:
                errors.append(
                    f"num_key_value_heads ({self.num_key_value_heads}) must not exceed "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            elif self.num_attention_heads % self.num_key_value_heads != 0:
                errors.append(
                    f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads})"
                )

        if (
            not isinstance(self.hidden_size, bool)
            and not isinstance(self.num_attention_heads, bool)
            and self.hidden_size > 0
            and self.num_attention_heads > 0
        ):
            if self.hidden_size % self.num_attention_heads != 0:
                errors.append(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            else:
                head_dim = self.hidden_size // self.num_attention_heads
                if head_dim % 2 != 0:
                    errors.append(f"head_dim ({head_dim}) must be even (divisible by 2) for RoPE")

        # Float checks
        if self.rope_theta <= 0.0:
            errors.append(f"rope_theta must be positive (> 0.0), got {self.rope_theta}")
        if self.rms_norm_eps <= 0.0:
            errors.append(f"rms_norm_eps must be positive (> 0.0), got {self.rms_norm_eps}")
        if not (0.0 <= self.attention_dropout <= 1.0):
            errors.append(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")
        if not (0.0 <= self.hidden_dropout <= 1.0):
            errors.append(f"hidden_dropout must be in [0, 1], got {self.hidden_dropout}")
        if self.initializer_range <= 0.0:
            errors.append(
                f"initializer_range must be positive (> 0.0), got {self.initializer_range}"
            )

        if errors:
            raise ValueError("BharatModelConfig validation failed:\n" + "\n".join(errors))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def gqa_ratio(self) -> int:
        return self.num_key_value_groups

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BharatModelConfig:
        if "architecture" in d and isinstance(d["architecture"], dict):
            return cls.from_dict(d["architecture"])
        known_keys = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "attention_dropout",
            "hidden_dropout",
            "initializer_range",
            "attention_bias",
            "mlp_bias",
            "tie_word_embeddings",
        }
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BharatModelConfig:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
