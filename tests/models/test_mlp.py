from __future__ import annotations

import pytest
import torch

from bharat.models.mlp import SwiGLU


class TestSwiGLU:
    @pytest.fixture
    def mlp(self):
        return SwiGLU(hidden_size=64, intermediate_size=256, bias=False, dropout=0.0)

    def test_output_shape(self, mlp: SwiGLU):
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)

    def test_forward_reference(self):
        mlp = SwiGLU(hidden_size=8, intermediate_size=16, bias=False, dropout=0.0)
        x = torch.randn(2, 4, 8)

        gate = torch.nn.functional.silu(mlp.gate_proj(x))
        up = mlp.up_proj(x)
        expected = mlp.down_proj(gate * up)

        out = mlp(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_backward_pass(self, mlp: SwiGLU):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_projections_get_gradients(self, mlp: SwiGLU):
        x = torch.randn(2, 8, 64)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None
        assert torch.isfinite(mlp.gate_proj.weight.grad).all()
        assert torch.isfinite(mlp.up_proj.weight.grad).all()
        assert torch.isfinite(mlp.down_proj.weight.grad).all()

    def test_dropout_disabled_in_eval(self):
        mlp = SwiGLU(hidden_size=32, intermediate_size=128, dropout=0.5)
        mlp.eval()
        x = torch.randn(4, 16, 32)
        out1 = mlp(x)
        out2 = mlp(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_active_in_train(self):
        mlp = SwiGLU(hidden_size=32, intermediate_size=128, dropout=0.5)
        mlp.train()
        x = torch.randn(4, 16, 32)
        out1 = mlp(x)
        out2 = mlp(x)
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_state_dict_roundtrip(self, mlp: SwiGLU):
        state = mlp.state_dict()
        keys = list(state.keys())
        assert "gate_proj.weight" in keys
        assert "up_proj.weight" in keys
        assert "down_proj.weight" in keys

        loaded = SwiGLU(hidden_size=64, intermediate_size=256, bias=False, dropout=0.0)
        loaded.load_state_dict(state)
        x = torch.randn(2, 8, 64)
        assert torch.allclose(mlp(x), loaded(x), atol=1e-6)

    def test_bias_option(self):
        mlp_bias = SwiGLU(hidden_size=32, intermediate_size=64, bias=True, dropout=0.0)
        assert mlp_bias.gate_proj.bias is not None
        assert mlp_bias.up_proj.bias is not None
        assert mlp_bias.down_proj.bias is not None

        mlp_no_bias = SwiGLU(hidden_size=32, intermediate_size=64, bias=False, dropout=0.0)
        assert mlp_no_bias.gate_proj.bias is None
        assert mlp_no_bias.up_proj.bias is None
        assert mlp_no_bias.down_proj.bias is None

    def test_silu_activation(self, mlp: SwiGLU):
        x = torch.zeros(1, 1, 64)
        out = mlp(x)
        assert out.shape == (1, 1, 64)
        assert torch.isfinite(out).all()
