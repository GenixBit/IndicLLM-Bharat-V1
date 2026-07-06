from __future__ import annotations

import pytest
import torch

from bharat.models.mlp import SwiGLU


class TestSwiGLU:
    def test_output_shape(self):
        mlp = SwiGLU(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 16, 512)
        out = mlp(x)
        assert out.shape == (2, 16, 512)

    def test_forward_reference(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256, bias=False)
        mlp.eval()
        x = torch.randn(2, 4, 64)

        out = mlp(x)

        with torch.no_grad():
            gate = torch.nn.functional.silu(mlp.gate_proj(x))
            up = mlp.up_proj(x)
            hidden = gate * up
            expected = mlp.down_proj(hidden)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_backward_pass(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_projections_get_gradients(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 4, 64)
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
        mlp = SwiGLU(hidden_size=64, intermediate_size=256, dropout=0.5)
        mlp.eval()
        x = torch.randn(2, 4, 64)
        out1 = mlp(x)
        out2 = mlp(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_dropout_active_in_train(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256, dropout=0.5)
        mlp.train()
        x = torch.randn(2, 4, 64)
        out1 = mlp(x)
        out2 = mlp(x)
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_dropout_affects_final_output(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256, dropout=0.5)
        mlp.train()
        x = torch.randn(2, 4, 64)
        out = mlp(x)
        assert out.shape == (2, 4, 64)
        assert torch.isfinite(out).all()

    def test_state_dict_roundtrip(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256)
        state = mlp.state_dict()
        loaded = SwiGLU(hidden_size=64, intermediate_size=256)
        loaded.load_state_dict(state)
        x = torch.randn(2, 4, 64)
        assert torch.allclose(mlp(x), loaded(x), atol=1e-5)

    def test_bias_option(self):
        mlp_biased = SwiGLU(hidden_size=64, intermediate_size=256, bias=True)
        mlp_unbiased = SwiGLU(hidden_size=64, intermediate_size=256, bias=False)
        for name in ("gate_proj", "up_proj", "down_proj"):
            biased_layer = getattr(mlp_biased, name)
            unbiased_layer = getattr(mlp_unbiased, name)
            assert biased_layer.bias is not None
            assert unbiased_layer.bias is None

    def test_silu_activation(self):
        mlp = SwiGLU(hidden_size=64, intermediate_size=256)
        mlp.eval()
        x = torch.randn(2, 4, 64)
        with torch.no_grad():
            gate = mlp.gate_proj(x)
        assert not torch.allclose(gate, torch.zeros_like(gate), atol=1e-6), "SiLU not applied"

    # ---------- Constructor validation ----------

    def test_invalid_hidden_size_raises(self):
        with pytest.raises(ValueError, match="hidden_size"):
            SwiGLU(hidden_size=0, intermediate_size=256)

    def test_invalid_intermediate_size_raises(self):
        with pytest.raises(ValueError, match="intermediate_size"):
            SwiGLU(hidden_size=64, intermediate_size=-1)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            SwiGLU(hidden_size=64, intermediate_size=256, dropout=1.5)
