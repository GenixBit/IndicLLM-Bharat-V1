from __future__ import annotations

import pytest
import torch

from bharat.models.normalization import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(hidden_size=512)
        x = torch.randn(2, 16, 512)
        out = norm(x)
        assert out.shape == (2, 16, 512)

    def test_reference_calculation(self):
        norm = RMSNorm(hidden_size=8, eps=1e-6)
        norm.weight.data = torch.ones(8) * 2.0
        x = torch.randn(2, 4, 8)
        out = norm(x)

        with torch.no_grad():
            variance = x.float().pow(2).mean(-1, keepdim=True)
            expected = x.float() * torch.rsqrt(variance + 1e-6)
            expected = (expected * 2.0).to(x.dtype)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_finite_outputs(self):
        norm = RMSNorm(hidden_size=512)
        x = torch.randn(2, 16, 512)
        out = norm(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_bfloat16_input(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

    def test_state_dict_roundtrip(self):
        norm = RMSNorm(hidden_size=64)
        state = norm.state_dict()
        loaded = RMSNorm(hidden_size=64)
        loaded.load_state_dict(state)
        x = torch.randn(2, 4, 64)
        assert torch.allclose(norm(x), loaded(x), atol=1e-5)

    def test_no_input_mutation(self):
        norm = RMSNorm(hidden_size=16)
        x = torch.randn(2, 8, 16)
        x_copy = x.clone()
        _out = norm(x)
        assert torch.allclose(x, x_copy, atol=1e-6)

    def test_2d_input(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(4, 64)
        out = norm(x)
        assert out.shape == (4, 64)

    def test_4d_input(self):
        norm = RMSNorm(hidden_size=32)
        x = torch.randn(2, 3, 4, 32)
        out = norm(x)
        assert out.shape == (2, 3, 4, 32)

    def test_learnable_scale(self):
        norm = RMSNorm(hidden_size=8, eps=1e-6)
        x = torch.randn(2, 4, 8)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert norm.weight.grad is not None
        assert not torch.allclose(norm.weight.grad, torch.zeros_like(norm.weight.grad), atol=1e-6)

    def test_weight_applied_in_bf16_module(self):
        norm = RMSNorm(hidden_size=64, eps=1e-6).to(dtype=torch.bfloat16)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    # ---------- Constructor validation ----------

    def test_invalid_hidden_size_raises(self):
        with pytest.raises(ValueError, match="hidden_size"):
            RMSNorm(hidden_size=0)

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError, match="eps"):
            RMSNorm(hidden_size=64, eps=0.0)
