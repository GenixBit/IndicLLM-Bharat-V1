from __future__ import annotations

import pytest
import torch

from bharat.models.normalization import RMSNorm


class TestRMSNorm:
    @pytest.fixture
    def rms(self):
        return RMSNorm(hidden_size=64, eps=1e-6)

    def test_output_shape(self, rms: RMSNorm):
        x = torch.randn(2, 16, 64)
        out = rms(x)
        assert out.shape == x.shape

    def test_reference_calculation(self):
        hidden_size = 8
        eps = 1e-6
        rms = RMSNorm(hidden_size=hidden_size, eps=eps)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        out = rms(x)

        mean_sq = x.float().pow(2).mean(-1, keepdim=True)
        expected = x.float() * torch.rsqrt(mean_sq + eps)
        expected = expected * rms.weight

        assert torch.allclose(out, expected, atol=1e-6)

    def test_finite_outputs(self, rms: RMSNorm):
        x = torch.randn(4, 32, 64)
        out = rms(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, rms: RMSNorm):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert rms.weight.grad is not None
        assert torch.isfinite(rms.weight.grad).all()

    def test_bfloat16_input(self):
        rms = RMSNorm(hidden_size=32, eps=1e-6).bfloat16()
        x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
        out = rms(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape
        assert torch.isfinite(out.float()).all()

    def test_state_dict_roundtrip(self, rms: RMSNorm):
        state = rms.state_dict()
        assert "weight" in state
        assert state["weight"].shape == (64,)

        loaded = RMSNorm(hidden_size=64, eps=1e-6)
        loaded.load_state_dict(state)
        x = torch.randn(2, 8, 64)
        assert torch.equal(rms(x), loaded(x))

    def test_no_input_mutation(self, rms: RMSNorm):
        x = torch.randn(2, 8, 64)
        x_copy = x.clone()
        _ = rms(x)
        assert torch.equal(x, x_copy)

    def test_2d_input(self):
        rms = RMSNorm(hidden_size=16, eps=1e-6)
        x = torch.randn(8, 16)
        out = rms(x)
        assert out.shape == (8, 16)

    def test_4d_input(self):
        rms = RMSNorm(hidden_size=32, eps=1e-6)
        x = torch.randn(2, 4, 8, 32)
        out = rms(x)
        assert out.shape == x.shape

    def test_learnable_scale(self):
        rms = RMSNorm(hidden_size=8, eps=1e-6)
        assert rms.weight.requires_grad
        x = torch.randn(4, 8)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert rms.weight.grad is not None
        assert not torch.equal(rms.weight.grad, torch.zeros_like(rms.weight.grad))
