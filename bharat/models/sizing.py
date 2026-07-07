from __future__ import annotations

import math
from dataclasses import dataclass

from bharat.models.config import BharatModelConfig


@dataclass(frozen=True)
class ParameterCount:
    token_embeddings: int
    attention_per_layer: int
    mlp_per_layer: int
    norms_per_layer: int
    transformer_layers: int
    final_norm: int
    lm_head: int
    total: int


def calculate_parameter_count(config: BharatModelConfig) -> ParameterCount:
    H = config.hidden_size
    I = config.intermediate_size
    L = config.num_hidden_layers
    A = config.num_attention_heads
    K = config.num_key_value_heads
    D = H // A
    V = config.vocab_size

    token_embeddings = V * H

    q_proj = H * H
    k_proj = H * (K * D)
    v_proj = H * (K * D)
    o_proj = H * H
    attention_weights = q_proj + k_proj + v_proj + o_proj
    if config.attention_bias:
        attention_weights += H + (K * D) + (K * D) + H

    gate_proj = H * I
    up_proj = H * I
    down_proj = I * H
    mlp_weights = gate_proj + up_proj + down_proj
    if config.mlp_bias:
        mlp_weights += I + I + H

    attention_per_layer = attention_weights
    mlp_per_layer = mlp_weights
    norms_per_layer = 2 * H

    transformer_layers = L * (attention_per_layer + mlp_per_layer + norms_per_layer)

    final_norm = H

    if config.tie_word_embeddings:
        lm_head = 0
    else:
        lm_head = V * H

    total = token_embeddings + transformer_layers + final_norm + lm_head

    return ParameterCount(
        token_embeddings=token_embeddings,
        attention_per_layer=attention_per_layer,
        mlp_per_layer=mlp_per_layer,
        norms_per_layer=norms_per_layer,
        transformer_layers=transformer_layers,
        final_norm=final_norm,
        lm_head=lm_head,
        total=total,
    )


# ---------------------------------------------------------------------------
# Memory calculators
# ---------------------------------------------------------------------------


_DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

_SUPPORTED_DTYPES: frozenset[str] = frozenset(_DTYPE_BYTES)


@dataclass(frozen=True)
class StaticMemoryReport:
    parameter_count: int
    weight_bytes: int
    gradient_bytes: int
    master_weight_bytes: int
    optimizer_state_bytes: int
    total_training_state_bytes: int


def calculate_static_memory(
    parameter_count: int,
    weight_dtype: str,
    gradient_dtype: str | None = None,
    optimizer: str | None = None,
    use_fp32_master_weights: bool = False,
) -> StaticMemoryReport:
    weight_dtype = weight_dtype.lower()
    if weight_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported weight dtype '{weight_dtype}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_DTYPES))}"
        )

    weight_bytes_per = _DTYPE_BYTES[weight_dtype]
    if weight_dtype == "int4":
        bits = parameter_count * 4
        weight_bytes = (bits + 7) // 8
    else:
        weight_bytes = _float_to_int(parameter_count * weight_bytes_per)

    if gradient_dtype is not None:
        grad_dtype = gradient_dtype.lower()
        if grad_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported gradient dtype '{grad_dtype}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_DTYPES))}"
            )
        grad_bytes_per = _DTYPE_BYTES[grad_dtype]
        if grad_dtype == "int4":
            gradient_bytes = (parameter_count * 4 + 7) // 8
        else:
            gradient_bytes = _float_to_int(parameter_count * grad_bytes_per)
    else:
        gradient_bytes = 0

    if use_fp32_master_weights:
        master_weight_bytes = _float_to_int(parameter_count * 4.0)
    else:
        master_weight_bytes = 0

    if optimizer == "adamw_fp32":
        optimizer_state_bytes = _float_to_int(parameter_count * 4.0 * 2)
    elif optimizer is None:
        optimizer_state_bytes = 0
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Supported: None, 'adamw_fp32'")

    total_training_state_bytes = (
        weight_bytes + gradient_bytes + master_weight_bytes + optimizer_state_bytes
    )

    return StaticMemoryReport(
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        gradient_bytes=gradient_bytes,
        master_weight_bytes=master_weight_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        total_training_state_bytes=total_training_state_bytes,
    )


@dataclass(frozen=True)
class KVCacheMemoryReport:
    bytes_per_token_per_batch_item: int
    total_bytes: int


def calculate_kv_cache_memory(
    config: BharatModelConfig,
    batch_size: int,
    sequence_length: int,
    dtype: str,
) -> KVCacheMemoryReport:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError(f"sequence_length must be an integer, got {type(sequence_length).__name__}")
    if sequence_length < 0:
        raise ValueError(f"sequence_length must be non-negative, got {sequence_length}")

    if sequence_length > config.max_position_embeddings:
        raise ValueError(
            f"sequence_length ({sequence_length}) exceeds "
            f"max_position_embeddings ({config.max_position_embeddings})"
        )

    dtype = dtype.lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported: {', '.join(sorted(_SUPPORTED_DTYPES))}"
        )

    bytes_per_elem = _DTYPE_BYTES[dtype]
    head_dim = config.hidden_size // config.num_attention_heads

    per_token = (
        config.num_hidden_layers * 2 * config.num_key_value_heads * head_dim * bytes_per_elem
    )
    total = batch_size * sequence_length * per_token

    return KVCacheMemoryReport(
        bytes_per_token_per_batch_item=_float_to_int(per_token),
        total_bytes=_float_to_int(total),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float_to_int(value: float) -> int:
    result = int(value)
    if result != value:
        return math.ceil(value)
    return result


def _ceil_mul(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m
