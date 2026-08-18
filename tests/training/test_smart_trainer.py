from __future__ import annotations

from pathlib import Path

from bharat.training.smart_trainer import (
    SmartTrainerConfig,
    get_tier_config,
    train_smart_bharat,
)


class TestSmartTrainer:
    def test_tier_configs(self):
        tiny = get_tier_config("tiny")
        assert tiny.hidden_size == 64
        assert tiny.num_hidden_layers == 2

        small = get_tier_config("small")
        assert small.hidden_size == 256
        assert small.num_hidden_layers == 4

        ten_b = get_tier_config("10b")
        assert ten_b.hidden_size == 4096
        assert ten_b.num_hidden_layers == 44

    def test_train_smart_bharat_tiny_cycle(self, tmp_path: Path):
        cfg = SmartTrainerConfig(
            model_tier="tiny",
            curriculum_dir=tmp_path / "curriculum",
            output_dir=tmp_path / "checkpoints",
            num_samples=10,
            pretrain_iters=4,
            sft_iters=2,
            batch_size=2,
            block_size=64,
            learning_rate=1e-3,
            device="cpu",
        )
        res = train_smart_bharat(cfg)
        assert res.model_tier == "tiny"
        assert Path(res.checkpoint_path).is_file()
        assert res.final_pretrain_loss > 0.0
