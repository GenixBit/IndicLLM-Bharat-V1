from __future__ import annotations

import torch

KeyValueCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[KeyValueCache, ...]


def past_length(past_key_values: PastKeyValues | None) -> int:
    """Return the total cached sequence length, or 0 if no cache is supplied."""
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    return past_key_values[0][0].shape[-2]


def validate_cache(
    cache: PastKeyValues,
    expected_layers: int,
    batch_size: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """
    Validate that a KV cache has consistent shapes and types across all layers.

    Raises ``ValueError`` on any mismatch.
    """
    if len(cache) != expected_layers:
        raise ValueError(f"Expected {expected_layers} cache layers, got {len(cache)}")

    cached_length: int | None = None
    for i, (k, v) in enumerate(cache):
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError(f"Cache layer {i} contains non-tensor entries")
        if k.device != device:
            raise ValueError(f"Cache layer {i} key is on {k.device}, expected {device}")
        if v.device != device:
            raise ValueError(f"Cache layer {i} value is on {v.device}, expected {device}")
        if k.dtype != dtype:
            raise ValueError(f"Cache layer {i} key dtype is {k.dtype}, expected {dtype}")
        if v.dtype != dtype:
            raise ValueError(f"Cache layer {i} value dtype is {v.dtype}, expected {dtype}")
        if k.dim() != 4:
            raise ValueError(f"Cache layer {i} key has {k.dim()} dimensions, expected 4")
        if v.dim() != 4:
            raise ValueError(f"Cache layer {i} value has {v.dim()} dimensions, expected 4")
        if k.shape[0] != batch_size:
            raise ValueError(f"Cache layer {i} key batch size {k.shape[0]} != {batch_size}")
        if v.shape[0] != batch_size:
            raise ValueError(f"Cache layer {i} value batch size {v.shape[0]} != {batch_size}")
        if k.shape[1] != kv_heads:
            raise ValueError(f"Cache layer {i} key has {k.shape[1]} heads, expected {kv_heads}")
        if v.shape[1] != kv_heads:
            raise ValueError(f"Cache layer {i} value has {v.shape[1]} heads, expected {kv_heads}")
        if k.shape[3] != head_dim:
            raise ValueError(f"Cache layer {i} key head_dim {k.shape[3]} != {head_dim}")
        if v.shape[3] != head_dim:
            raise ValueError(f"Cache layer {i} value head_dim {v.shape[3]} != {head_dim}")
        if k.shape != v.shape:
            raise ValueError(f"Cache layer {i} key shape {k.shape} != value shape {v.shape}")
        if cached_length is None:
            cached_length = k.shape[2]
        elif k.shape[2] != cached_length:
            raise ValueError(
                f"Cache layer {i} has inconsistent length {k.shape[2]} (expected {cached_length})"
            )


def reorder_cache(
    cache: PastKeyValues,
    indices: torch.Tensor,
) -> PastKeyValues:
    """
    Reorder the batch dimension of all cache layers according to ``indices``.

    ``indices`` must be a 1-D integer tensor of length ``batch_size`` with
    values in ``[0, batch_size)``.

    When ``indices`` is empty, the cache is returned unchanged (no elements
    to select).  The original cache is never mutated.

    Useful for future beam-search or sorting operations.
    """
    if indices.dim() != 1:
        raise ValueError(f"reorder_cache indices must be 1-D, got {indices.dim()}-D")
    if indices.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
        raise ValueError(f"reorder_cache indices must be an integer dtype, got {indices.dtype}")
    if len(indices) > 0:
        batch_size = cache[0][0].shape[0]
        if indices.min() < 0 or indices.max() >= batch_size:
            raise ValueError(
                f"reorder_cache indices must be in [0, {batch_size - 1}], "
                f"got range [{indices.min().item()}, {indices.max().item()}]"
            )
    return tuple((k.index_select(0, indices), v.index_select(0, indices)) for k, v in cache)
