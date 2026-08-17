from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.posttraining.dpo import DPOConfig, dpo_train
from bharat.posttraining.preference_loss import per_sample_log_probs
from bharat.posttraining.sft import SFTConfig, sft_train


def make_bharat_config(vocab_size: int = 512) -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        initializer_range=0.02,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )


@pytest.fixture
def sft_data_file(tmp_path: Path) -> Path:
    data_path = tmp_path / "sft_data.jsonl"
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "I am fine thank you"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France"},
                {"role": "assistant", "content": "Paris is the capital of France"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me about AI"},
                {"role": "assistant", "content": "Machine learning is fascinating"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain tokens"},
                {"role": "assistant", "content": "Tokenization breaks text into pieces"},
            ]
        },
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return data_path


@pytest.fixture
def dpo_data_file(tmp_path: Path) -> Path:
    data_path = tmp_path / "dpo_data.jsonl"
    samples = [
        {
            "prompt": "What is France capital",
            "chosen": "Paris is the capital of France",
            "rejected": "Hello world today",
        },
        {
            "prompt": "Explain machine learning",
            "chosen": "Machine learning is fascinating",
            "rejected": "I am fine thank you",
        },
        {
            "prompt": "What are tokens",
            "chosen": "Tokenization breaks text into pieces",
            "rejected": "a b c d e f g",
        },
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return data_path


class TestModernPostTrainingIntegration:
    def test_sft_train_with_bharat_model(
        self,
        tmp_path: Path,
        sft_data_file: Path,
        tiny_tokenizer,
    ) -> None:
        """Verify that SFT training executes on BharatForCausalLM with loss calculation and checkpointing."""
        torch.manual_seed(42)
        cfg_model = make_bharat_config(vocab_size=tiny_tokenizer.vocab_size)
        model = BharatForCausalLM(cfg_model)

        cfg = SFTConfig(
            data_path=sft_data_file,
            output_dir=tmp_path / "sft_out",
            max_iters=15,
            batch_size=2,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=2,
            log_interval=5,
            save_interval=5,
            device="cpu",
        )

        result = sft_train(model=model, config=cfg, tokenizer=tiny_tokenizer)

        assert result.completed_steps == 15
        assert result.final_loss < float("inf")
        assert result.samples_processed > 0
        assert result.active_tokens > 0
        assert (tmp_path / "sft_out" / "final.pt").is_file()
        assert (tmp_path / "sft_out" / "best.pt").is_file()

    def test_per_sample_log_probs_with_bharat_causal_lm(self, tiny_tokenizer) -> None:
        """Verify per_sample_log_probs works with BharatForCausalLM outputs and propagates gradients."""
        torch.manual_seed(42)
        cfg_model = make_bharat_config(vocab_size=tiny_tokenizer.vocab_size)
        model = BharatForCausalLM(cfg_model)
        model.train()

        input_ids = torch.randint(0, tiny_tokenizer.vocab_size, (2, 16))
        # Mask out first 5 tokens (prompt), keep last 10 (response targets)
        response_mask = torch.zeros((2, 15), dtype=torch.bool)
        response_mask[:, 5:] = True

        lp = per_sample_log_probs(
            model=model,
            input_ids=input_ids,
            response_masks=response_mask,
            ctx=torch.enable_grad(),
        )

        assert lp.shape == (2,)
        assert not torch.isnan(lp).any()
        assert not torch.isinf(lp).any()

        # Check backward pass
        loss = -lp.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} has no grad"

    def test_dpo_train_with_bharat_model(
        self,
        tmp_path: Path,
        dpo_data_file: Path,
        tiny_tokenizer,
    ) -> None:
        """Verify that DPO training executes on BharatForCausalLM policy and reference models."""
        torch.manual_seed(42)
        cfg_model = make_bharat_config(vocab_size=tiny_tokenizer.vocab_size)
        policy_model = BharatForCausalLM(cfg_model)
        ref_model = BharatForCausalLM(cfg_model)

        cfg = DPOConfig(
            data_path=dpo_data_file,
            output_dir=tmp_path / "dpo_out",
            max_iters=10,
            batch_size=2,
            block_size=64,
            learning_rate=1e-3,
            beta=0.1,
            log_interval=5,
            save_interval=5,
            device="cpu",
        )

        result = dpo_train(
            policy_model=policy_model,
            ref_model=ref_model,
            config=cfg,
            tokenizer=tiny_tokenizer,
        )

        assert result.completed_steps == 10
        assert result.final_loss < float("inf")
        assert result.samples_processed > 0
        assert (tmp_path / "dpo_out" / "final.pt").is_file()
        assert (tmp_path / "dpo_out" / "best.pt").is_file()
