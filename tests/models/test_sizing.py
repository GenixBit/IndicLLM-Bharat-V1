from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.models.sizing import (
    KVCacheMemoryReport,
    calculate_kv_cache_memory,
    calculate_parameter_count,
    calculate_static_memory,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def _small_config(
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 256,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 64,
    tie_word_embeddings: bool = True,
    attention_bias: bool = False,
    mlp_bias: bool = False,
) -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
    )


def _finites(p: torch.Tensor) -> int:
    return int(torch.isfinite(p).sum().item())


def test_calculator_matches_real_model_mha():
    cfg = _small_config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total, f"Analytical {report.total} != real {total}"


def test_calculator_matches_real_model_gqa():
    cfg = _small_config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total, f"Analytical {report.total} != real {total}"


def test_calculator_matches_real_model_mqa():
    cfg = _small_config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
    )
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_calculator_matches_real_model_untied():
    cfg = _small_config(tie_word_embeddings=False)
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_calculator_matches_with_attention_bias():
    cfg = _small_config(attention_bias=True)
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_calculator_matches_with_mlp_bias():
    cfg = _small_config(mlp_bias=True)
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_calculator_matches_with_both_biases():
    cfg = _small_config(attention_bias=True, mlp_bias=True)
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_calculator_matches_different_vocab():
    cfg = _small_config(vocab_size=256)
    report = calculate_parameter_count(cfg)
    model = BharatForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert report.total == total


def test_untied_embeddings_add_lm_head():
    tied_cfg = _small_config(tie_word_embeddings=True)
    untied_cfg = _small_config(tie_word_embeddings=False, vocab_size=128, hidden_size=64)
    tied_params = calculate_parameter_count(tied_cfg)
    untied_params = calculate_parameter_count(untied_cfg)
    expected_extra = 128 * 64
    assert untied_params.lm_head == expected_extra
    assert untied_params.total == tied_params.total + expected_extra


def test_attention_bias_adds_correct_params():
    nobias_cfg = _small_config(
        attention_bias=False,
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    bias_cfg = _small_config(
        attention_bias=True,
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    nobias_report = calculate_parameter_count(nobias_cfg)
    bias_report = calculate_parameter_count(bias_cfg)
    h, k, d = 32, 4, 8
    expected_bias = h + (k * d) + (k * d) + h  # Q, K, V, O
    assert bias_report.attention_per_layer - nobias_report.attention_per_layer == expected_bias


def test_mlp_bias_adds_correct_params():
    nobias_cfg = _small_config(
        mlp_bias=False,
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    bias_cfg = _small_config(
        mlp_bias=True,
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    nobias_report = calculate_parameter_count(nobias_cfg)
    bias_report = calculate_parameter_count(bias_cfg)
    i = 64
    h = 32
    expected_bias = i + i + h
    assert bias_report.mlp_per_layer - nobias_report.mlp_per_layer == expected_bias


class TestProductionSpecTotals:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("bharat-350m.yaml", 347393024),
            ("bharat-1b.yaml", 999368704),
            ("bharat-3b.yaml", 3009039360),
            ("bharat-7b.yaml", 7040405504),
        ],
    )
    def test_exact_total(self, filename: str, expected: int) -> None:
        from bharat.models.spec import load_model_spec

        spec = load_model_spec(ROOT / "configs" / "models" / filename)
        params = calculate_parameter_count(spec.architecture)
        assert params.total == expected, (
            f"{filename}: analytical {params.total} != expected {expected}"
        )


class TestStaticMemory:
    def test_bf16_weight_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="bf16")
        assert report.weight_bytes == 2000
        assert report.parameter_count == 1000
        assert report.gradient_bytes == 0
        assert report.master_weight_bytes == 0
        assert report.optimizer_state_bytes == 0

    def test_fp32_weight_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="fp32")
        assert report.weight_bytes == 4000

    def test_fp16_weight_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="fp16")
        assert report.weight_bytes == 2000

    def test_int8_weight_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="int8")
        assert report.weight_bytes == 1000

    def test_int4_weight_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="int4")
        assert report.weight_bytes == 500  # 1000 * 0.5

    def test_gradient_bytes(self):
        report = calculate_static_memory(1000, weight_dtype="bf16", gradient_dtype="bf16")
        assert report.gradient_bytes == 2000

    def test_gradient_fp32(self):
        report = calculate_static_memory(1000, weight_dtype="bf16", gradient_dtype="fp32")
        assert report.gradient_bytes == 4000

    def test_adamw_optimizer(self):
        report = calculate_static_memory(1000, weight_dtype="bf16", optimizer="adamw_fp32")
        assert report.optimizer_state_bytes == 8000  # 1000 * 4 * 2

    def test_fp32_master_weights(self):
        report = calculate_static_memory(1000, weight_dtype="bf16", use_fp32_master_weights=True)
        assert report.master_weight_bytes == 4000

    def test_full_training_state(self):
        report = calculate_static_memory(
            1000,
            weight_dtype="bf16",
            gradient_dtype="bf16",
            optimizer="adamw_fp32",
            use_fp32_master_weights=True,
        )
        expected = 2000 + 2000 + 4000 + 8000
        assert report.total_training_state_bytes == expected

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            calculate_static_memory(1000, weight_dtype="fp64")

    def test_unsupported_optimizer_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            calculate_static_memory(1000, weight_dtype="bf16", optimizer="sgd")

    def test_int4_odd_count(self):
        report = calculate_static_memory(3, weight_dtype="int4")
        assert report.weight_bytes == 2  # ceil(3*0.5) = 2


class TestKVCacheMemory:
    def test_basic_gqa(self):
        cfg = _small_config(num_key_value_heads=4)
        report = calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=64, dtype="bf16")
        assert isinstance(report, KVCacheMemoryReport)
        assert report.total_bytes > 0

    def test_mha_matches_gqa(self):
        cfg = _small_config(
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256
        )
        report = calculate_kv_cache_memory(cfg, batch_size=2, sequence_length=128, dtype="fp32")
        assert isinstance(report, KVCacheMemoryReport)
        assert report.total_bytes > 0

    def test_mqa_less_than_mha(self):
        mha = _small_config(
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256
        )
        mqa = _small_config(
            num_attention_heads=4, num_key_value_heads=1, max_position_embeddings=256
        )
        mha_report = calculate_kv_cache_memory(mha, batch_size=1, sequence_length=64, dtype="bf16")
        mqa_report = calculate_kv_cache_memory(mqa, batch_size=1, sequence_length=64, dtype="bf16")
        assert mqa_report.total_bytes < mha_report.total_bytes

    def test_proportional_to_batch(self):
        cfg = _small_config(max_position_embeddings=256)
        r1 = calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=128, dtype="fp32")
        r2 = calculate_kv_cache_memory(cfg, batch_size=2, sequence_length=128, dtype="fp32")
        assert r2.total_bytes == 2 * r1.total_bytes

    def test_proportional_to_seq(self):
        cfg = _small_config(max_position_embeddings=256)
        r1 = calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=64, dtype="fp32")
        r2 = calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=128, dtype="fp32")
        assert r2.total_bytes == 2 * r1.total_bytes

    def test_uses_kv_heads_not_query_heads(self):
        cfg = _small_config(
            num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=256
        )
        report = calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=1, dtype="fp32")
        h = cfg.hidden_size
        d = h // cfg.num_attention_heads
        expected = 1 * 1 * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * d * 4
        assert report.total_bytes == expected

    def test_invalid_batch_size_raises(self):
        cfg = _small_config()
        with pytest.raises((ValueError, TypeError)):
            calculate_kv_cache_memory(cfg, batch_size=0, sequence_length=1, dtype="fp32")

    def test_context_overflow_raises(self):
        cfg = _small_config(max_position_embeddings=64)
        with pytest.raises(ValueError, match="exceeds"):
            calculate_kv_cache_memory(cfg, batch_size=1, sequence_length=128, dtype="fp32")

    def test_bool_batch_size_raises(self):
        cfg = _small_config()
        with pytest.raises(TypeError, match="batch_size"):
            calculate_kv_cache_memory(cfg, batch_size=True, sequence_length=1, dtype="fp32")


class TestProductionMemory:
    @pytest.mark.parametrize(
        "filename,expected_bytes",
        [
            ("bharat-350m.yaml", 347393024 * 2),
            ("bharat-1b.yaml", 999368704 * 2),
            ("bharat-3b.yaml", 3009039360 * 2),
            ("bharat-7b.yaml", 7040405504 * 2),
        ],
    )
    def test_bf16_weight_bytes(self, filename: str, expected_bytes: int) -> None:
        from bharat.models.spec import load_model_spec

        spec = load_model_spec(ROOT / "configs" / "models" / filename)
        params = calculate_parameter_count(spec.architecture)
        report = calculate_static_memory(params.total, weight_dtype="bf16")
        assert report.weight_bytes == expected_bytes

    @pytest.mark.parametrize(
        "filename,expected_mib",
        [
            ("bharat-350m.yaml", 100.0),
            ("bharat-1b.yaml", 144.0),
            ("bharat-3b.yaml", 384.0),
            ("bharat-7b.yaml", 512.0),
        ],
    )
    def test_kv_cache_bf16_bs1_seq4096(self, filename: str, expected_mib: float) -> None:
        from bharat.models.spec import load_model_spec

        spec = load_model_spec(ROOT / "configs" / "models" / filename)
        report = calculate_kv_cache_memory(
            spec.architecture, batch_size=1, sequence_length=4096, dtype="bf16"
        )
        actual_mib = report.total_bytes / (1024**2)
        assert actual_mib == pytest.approx(expected_mib, rel=1e-4), (
            f"{filename}: expected {expected_mib} MiB, got {actual_mib} MiB"
        )
