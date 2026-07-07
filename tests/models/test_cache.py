from __future__ import annotations

import pytest
import torch

from bharat.models.cache import (
    PastKeyValues,
    past_length,
    reorder_cache,
    validate_cache,
)


def _make_cache(
    num_layers: int = 2,
    batch_size: int = 2,
    kv_heads: int = 4,
    seq_len: int = 5,
    head_dim: int = 64,
) -> PastKeyValues:
    return tuple(
        (
            torch.randn(batch_size, kv_heads, seq_len, head_dim),
            torch.randn(batch_size, kv_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )


class TestPastLength:
    def test_none_returns_zero(self):
        assert past_length(None) == 0

    def test_empty_tuple_returns_zero(self):
        assert past_length(()) == 0

    def test_returns_cached_length(self):
        cache = _make_cache(seq_len=7)
        assert past_length(cache) == 7


class TestValidateCache:
    def test_valid_cache_passes(self):
        cache = _make_cache(num_layers=2, batch_size=2, kv_heads=4, seq_len=5, head_dim=64)
        validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_wrong_layer_count_raises(self):
        cache = _make_cache(num_layers=3, batch_size=2, kv_heads=4, seq_len=5, head_dim=64)
        with pytest.raises(ValueError, match="Expected 2"):
            validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_wrong_batch_size_raises(self):
        cache = _make_cache(batch_size=3)
        with pytest.raises(ValueError, match="batch size"):
            validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_wrong_kv_heads_raises(self):
        cache = _make_cache(kv_heads=8)
        with pytest.raises(ValueError, match="heads"):
            validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_wrong_head_dim_raises(self):
        cache = _make_cache(head_dim=128)
        with pytest.raises(ValueError, match="head_dim"):
            validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_inconsistent_length_raises(self):
        cache = [
            (torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 64)),
            (torch.randn(2, 4, 7, 64), torch.randn(2, 4, 7, 64)),
        ]
        with pytest.raises(ValueError, match="inconsistent length"):
            validate_cache(tuple(cache), 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_device_mismatch_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cache = _make_cache()
        wrong_device = torch.device("cuda:0")
        with pytest.raises(ValueError, match="device"):
            validate_cache(cache, 2, 2, 4, 64, wrong_device, cache[0][0].dtype)

    def test_dtype_mismatch_raises(self):
        cache = _make_cache()
        wrong_dtype = torch.bfloat16
        if cache[0][0].dtype == torch.bfloat16:
            wrong_dtype = torch.float16
        with pytest.raises(ValueError, match="dtype"):
            validate_cache(cache, 2, 2, 4, 64, cache[0][0].device, wrong_dtype)

    def test_non_tensor_raises(self):
        cache = (("not_a_tensor", "not_a_tensor"),)
        with pytest.raises(ValueError, match="non-tensor"):
            validate_cache(tuple(cache), 1, 2, 4, 64, torch.device("cpu"), torch.float32)

    def test_key_value_shape_mismatch_raises(self):
        cache = [
            (torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 64)),
            (torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 32)),
        ]
        with pytest.raises(ValueError, match="value head_dim"):
            validate_cache(tuple(cache), 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_key_value_shape_mismatch_length(self):
        cache = [
            (torch.randn(2, 4, 5, 64), torch.randn(2, 4, 5, 64)),
            (torch.randn(2, 4, 5, 64), torch.randn(2, 4, 6, 64)),
        ]
        with pytest.raises(ValueError, match="key shape"):
            validate_cache(tuple(cache), 2, 2, 4, 64, cache[0][0].device, cache[0][0].dtype)

    def test_wrong_dimensions_raises(self):
        cache = [(torch.randn(2, 4, 5, 64, 1), torch.randn(2, 4, 5, 64, 1))]
        with pytest.raises(ValueError, match="4"):
            validate_cache(tuple(cache), 1, 2, 4, 64, torch.device("cpu"), torch.float32)


class TestReorderCache:
    def test_reorder_batch(self):
        cache = _make_cache(num_layers=2, batch_size=3, kv_heads=4, seq_len=5, head_dim=64)
        indices = torch.tensor([2, 0, 1])
        reordered = reorder_cache(cache, indices)
        assert len(reordered) == 2
        for i in range(2):
            assert torch.equal(reordered[i][0], cache[i][0][indices])
            assert torch.equal(reordered[i][1], cache[i][1][indices])

    def test_reorder_2d_indices_raises(self):
        cache = _make_cache(batch_size=3)
        with pytest.raises(ValueError, match="1-D"):
            reorder_cache(cache, torch.tensor([[0, 1], [2, 0]]))

    def test_reorder_float_indices_raises(self):
        cache = _make_cache(batch_size=3)
        with pytest.raises(ValueError, match="integer dtype"):
            reorder_cache(cache, torch.tensor([0.0, 1.0, 2.0]))

    def test_reorder_out_of_range_raises(self):
        cache = _make_cache(batch_size=3)
        with pytest.raises(ValueError, match="range"):
            reorder_cache(cache, torch.tensor([0, 1, 5]))

    def test_reorder_negative_index_raises(self):
        cache = _make_cache(batch_size=3)
        with pytest.raises(ValueError, match="range"):
            reorder_cache(cache, torch.tensor([0, -1, 1]))

    def test_reorder_empty_indices(self):
        cache = _make_cache(batch_size=3)
        indices = torch.tensor([], dtype=torch.long)
        reordered = reorder_cache(cache, indices)
        assert len(reordered) == len(cache)
        for i in range(len(cache)):
            assert reordered[i][0].shape[0] == 0
            assert reordered[i][1].shape[0] == 0

    def test_reorder_does_not_mutate_original(self):
        cache = _make_cache(batch_size=3)
        original_first = cache[0][0].clone()
        _ = reorder_cache(cache, torch.tensor([2, 0, 1]))
        assert torch.equal(cache[0][0], original_first), "Original cache was mutated"
