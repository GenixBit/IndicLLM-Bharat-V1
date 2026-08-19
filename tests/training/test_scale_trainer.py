from __future__ import annotations

from pathlib import Path

from bharat.data.world_knowledge import pack_world_knowledge_shards
from bharat.tokenizer import load_tokenizer
from bharat.training.scale_trainer import (
    BharatScaleTrainer,
    ScaleTrainerConfig,
    get_scale_tier_config,
)
from scripts.train_scale_bharat import main as train_scale_main
from scripts.train_scale_bharat import parse_args


class TestScaleTrainer:
    def test_tier_configs_validation(self):
        t1b = get_scale_tier_config("1b")
        assert t1b.hidden_size == 2048
        assert t1b.num_hidden_layers == 18
        assert t1b.num_attention_heads == 16
        assert t1b.num_key_value_heads == 4

        t3b = get_scale_tier_config("3b")
        assert t3b.hidden_size == 3072
        assert t3b.num_hidden_layers == 28

        t7b = get_scale_tier_config("7b")
        assert t7b.hidden_size == 4096
        assert t7b.num_hidden_layers == 32

        t10b = get_scale_tier_config("10b")
        assert t10b.hidden_size == 4096
        assert t10b.num_hidden_layers == 44
        assert t10b.intermediate_size == 14336

    def test_scale_trainer_tiny_step(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        tok = load_tokenizer("gpt2")
        pack_world_knowledge_shards(tok, shards_dir)

        out_dir = tmp_path / "ckpt"
        config = ScaleTrainerConfig(
            tier="tiny",
            shards_dir=shards_dir,
            output_dir=out_dir,
            steps=3,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            device="cpu",
            seed=42,
        )

        trainer = BharatScaleTrainer(config)
        res = trainer.train()

        assert res.tier == "tiny"
        assert res.parameter_count > 0
        assert res.final_loss > 0.0
        assert res.total_tokens_processed == 3 * 64
        assert Path(res.checkpoint_path).is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "10b", "--steps", "10", "--dry-run-calc"])
        assert args.tier == "10b"
        assert args.steps == 10
        assert args.dry_run_calc is True

    def test_cli_main_dry_run(self):
        code = train_scale_main(["--tier", "10b", "--dry-run-calc"])
        assert code == 0

    def test_cli_main_train(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        tok = load_tokenizer("gpt2")
        pack_world_knowledge_shards(tok, shards_dir)
        out_dir = tmp_path / "scale_out"

        code = train_scale_main(
            [
                "--tier",
                "tiny",
                "--steps",
                "2",
                "--shards-dir",
                str(shards_dir),
                "--output-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
        )
        assert code == 0
        assert (out_dir / "bharat_tiny" / "final.pt").is_file()
