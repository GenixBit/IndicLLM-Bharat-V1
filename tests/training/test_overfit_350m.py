from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.training.checkpointing import load_checkpoint, save_checkpoint
from bharat.training.pretrain import (
    PretrainConfig,
    configure_optimizers,
    get_cosine_lr,
    pretrain,
)
from scripts.pretrain_bharat import main as cli_main


@pytest.fixture
def tiny_bharat_config() -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=256,
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


class TestOverfitAndTrainingDynamics:
    def test_overfit_single_batch(self, tiny_bharat_config: BharatModelConfig) -> None:
        """
        Milestone 6.2 Requirement: Prove that the modern Bharat architecture
        (GQA, RoPE, RMSNorm, SwiGLU) can overfit a single fixed batch of tokens
        and drive cross-entropy loss from ~ln(V) down to < 0.1.
        """
        torch.manual_seed(42)
        model = BharatForCausalLM(tiny_bharat_config)
        model.train()

        batch_size = 2
        seq_len = 32
        vocab_size = tiny_bharat_config.vocab_size

        # Create a single fixed synthetic batch
        rng = np.random.RandomState(42)
        data = rng.randint(0, vocab_size, size=(batch_size, seq_len + 1))
        x = torch.from_numpy(data[:, :-1].astype(np.int64))
        y = torch.from_numpy(data[:, 1:].astype(np.int64))

        # Initial loss should be around ln(vocab_size) = ln(256) ≈ 5.54
        with torch.no_grad():
            init_out = model(input_ids=x, labels=y)
            init_loss = init_out.loss.item()
            assert init_loss > 3.0, f"Expected initial loss > 3.0, got {init_loss}"

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95))

        for _step in range(120):
            optimizer.zero_grad()
            out = model(input_ids=x, labels=y)
            loss = out.loss
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final_loss = loss.item()
        assert (
            final_loss < 0.1
        ), f"Overfit test failed: expected loss < 0.1, got {final_loss:.4f} after 120 steps"

    def test_all_parameters_receive_gradients(self, tiny_bharat_config: BharatModelConfig) -> None:
        """Verify that all active parameters receive non-zero gradients on a backward pass."""
        torch.manual_seed(42)
        model = BharatForCausalLM(tiny_bharat_config)
        model.train()

        x = torch.randint(0, tiny_bharat_config.vocab_size, (2, 16))
        y = torch.randint(0, tiny_bharat_config.vocab_size, (2, 16))

        out = model(input_ids=x, labels=y)
        assert out.loss is not None
        out.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                grad_norm = param.grad.norm().item()
                assert grad_norm > 0.0, f"Parameter {name} gradient is all zero"

    def test_configure_optimizers_parameter_partitioning(
        self, tiny_bharat_config: BharatModelConfig
    ) -> None:
        """Check that 2D weights get weight decay and 1D norms/biases get 0 weight decay."""
        model = BharatForCausalLM(tiny_bharat_config)
        optimizer = configure_optimizers(
            model,
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )

        assert len(optimizer.param_groups) == 2
        decay_group = optimizer.param_groups[0]
        nodecay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert nodecay_group["weight_decay"] == 0.0

        for p in decay_group["params"]:
            assert p.dim() >= 2
        for p in nodecay_group["params"]:
            assert p.dim() < 2

    def test_cosine_lr_schedule(self, tiny_bharat_config: BharatModelConfig) -> None:
        """Verify linear warmup and cosine decay curve."""
        cfg = PretrainConfig(
            model_config=tiny_bharat_config,
            max_iters=100,
            warmup_iters=20,
            learning_rate=1e-3,
            min_lr=1e-4,
        )

        # Step 0 (start of warmup)
        lr_0 = get_cosine_lr(0, cfg)
        assert lr_0 == pytest.approx(1e-3 * 1 / 20)

        # Step 19 (end of warmup)
        lr_19 = get_cosine_lr(19, cfg)
        assert lr_19 == pytest.approx(1e-3)

        # Step 60 (halfway through cosine decay)
        lr_60 = get_cosine_lr(60, cfg)
        expected_mid = 1e-4 + 0.5 * (1.0 + np.cos(np.pi * (60 - 20) / (100 - 20))) * (1e-3 - 1e-4)
        assert lr_60 == pytest.approx(expected_mid)

        # Step 100+ (post max-iters)
        lr_100 = get_cosine_lr(100, cfg)
        assert lr_100 == pytest.approx(1e-4)

    def test_checkpoint_save_and_resume_exact_loss(
        self, tmp_path: Path, tiny_bharat_config: BharatModelConfig
    ) -> None:
        """Verify that saving and loading training checkpoints reproduces identical training loss."""
        torch.manual_seed(42)
        model1 = BharatForCausalLM(tiny_bharat_config)
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

        x = torch.randint(0, tiny_bharat_config.vocab_size, (2, 16))
        y = torch.randint(0, tiny_bharat_config.vocab_size, (2, 16))

        # Train 5 steps
        for _ in range(5):
            opt1.zero_grad()
            out = model1(input_ids=x, labels=y)
            out.loss.backward()
            opt1.step()

        # Save checkpoint
        ckpt_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model1,
            optimizer=opt1,
            config=tiny_bharat_config.to_dict(),
            step=5,
            seed=42,
        )

        # Take 1 more step with model1
        opt1.zero_grad()
        out1 = model1(input_ids=x, labels=y)
        loss1 = out1.loss.item()

        # Load into model2
        model2 = BharatForCausalLM(tiny_bharat_config)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        load_checkpoint(ckpt_path, model2, optimizer=opt2, device="cpu")

        # Take 1 step with model2
        opt2.zero_grad()
        out2 = model2(input_ids=x, labels=y)
        loss2 = out2.loss.item()

        assert loss1 == pytest.approx(loss2, rel=1e-5)

    def test_pretrain_runner_synthetic(
        self, tmp_path: Path, tiny_bharat_config: BharatModelConfig
    ) -> None:
        """Run full pretrain orchestrator on synthetic stream for 15 iterations."""
        cfg = PretrainConfig(
            model_config=tiny_bharat_config,
            synthetic_data=True,
            output_dir=tmp_path / "checkpoints",
            max_iters=15,
            batch_size=2,
            seq_len=16,
            learning_rate=1e-3,
            warmup_iters=5,
            eval_interval=5,
            eval_iters=2,
            save_interval=10,
            seed=42,
        )

        result = pretrain(cfg)
        assert result.completed_steps == 15
        assert len(result.step_losses) == 15
        assert result.val_loss is not None
        assert result.checkpoint_path is not None
        assert Path(result.checkpoint_path).is_file()

    def test_pretrain_cli_smoke(self, tmp_path: Path) -> None:
        """Smoke test for CLI entrypoint."""
        config_path = tmp_path / "tiny_config.yaml"
        tiny_dict = {
            "schema_version": 1,
            "model_name": "Bharat-Tiny",
            "architecture": {
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "initializer_range": 0.02,
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(tiny_dict, f)

        args = [
            "--config",
            str(config_path),
            "--max-iters",
            "5",
            "--batch-size",
            "2",
            "--seq-len",
            "16",
            "--synthetic-data",
            "--output-dir",
            str(tmp_path / "cli_ckpt"),
            "--eval-interval",
            "5",
            "--eval-iters",
            "1",
            "--save-interval",
            "5",
            "--json",
        ]

        ret = cli_main(args)
        assert ret == 0
