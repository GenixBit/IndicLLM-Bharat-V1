from __future__ import annotations

from pathlib import Path

from bharat.data.preference_curriculum import export_preference_curriculum
from bharat.training.dpo_trainer import (
    BharatDPOTrainer,
    DPOTrainerConfig,
    build_dpo_sequence,
)
from scripts.train_dpo_bharat import main as train_dpo_main
from scripts.train_dpo_bharat import parse_args


class TestDPOTrainer:
    def test_parse_args(self):
        args = parse_args(
            [
                "--max-iters",
                "15",
                "--batch-size",
                "4",
                "--beta",
                "0.2",
                "--device",
                "cpu",
            ]
        )
        assert args.max_iters == 15
        assert args.batch_size == 4
        assert args.beta == 0.2
        assert args.device == "cpu"

    def test_build_dpo_sequence(self):
        from bharat.tokenizer import load_tokenizer

        tok = load_tokenizer("gpt2")
        ids, mask = build_dpo_sequence(tok, "Hello Bharat", "Hello! I am Bharat AI.", block_size=64)
        assert len(ids) > 0
        assert len(mask) == len(ids)
        assert any(mask)  # contains response tokens marked True

    def test_dpo_trainer_small_run(self, tmp_path: Path):
        data_path = tmp_path / "dpo_test.jsonl"
        export_preference_curriculum(data_path)

        out_dir = tmp_path / "checkpoints_dpo"

        config = DPOTrainerConfig(
            sft_checkpoint=tmp_path / "dummy.pt",  # fallback initializes fresh
            preference_data=data_path,
            output_dir=out_dir,
            model_tier="tiny",
            max_iters=4,
            batch_size=2,
            block_size=128,
            learning_rate=1e-4,
            beta=0.1,
            warmup_iters=1,
            device="cpu",
            seed=42,
        )

        trainer = BharatDPOTrainer(config)
        res = trainer.train()

        assert res.completed_steps == 4
        assert res.final_loss > 0.0
        assert 0.0 <= res.final_reward_accuracy <= 1.0
        assert Path(res.checkpoint_path).is_file()

    def test_cli_train_dpo_main(self, tmp_path: Path):
        data_path = tmp_path / "prefs.jsonl"
        export_preference_curriculum(data_path)
        out_dir = tmp_path / "ckpt"

        code = train_dpo_main(
            [
                "--preference-data",
                str(data_path),
                "--output-dir",
                str(out_dir),
                "--model-tier",
                "tiny",
                "--max-iters",
                "2",
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
        )
        assert code == 0
        assert (out_dir / "final.pt").is_file()
