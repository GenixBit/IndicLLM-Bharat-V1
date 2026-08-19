from __future__ import annotations

from pathlib import Path

from bharat.data.instruction_curriculum import export_instruction_curriculum
from bharat.posttraining.sft_trainer import (
    BharatSFTTrainer,
    SFTTrainingConfig,
)
from scripts.train_sft_bharat import main as train_sft_main
from scripts.train_sft_bharat import parse_args


class TestBharatSFTTrainer:
    def test_sft_dataset_preparation(self, tmp_path: Path):
        data_p = tmp_path / "sft_data.jsonl"
        export_instruction_curriculum(data_p)

        config = SFTTrainingConfig(
            tier="tiny",
            data_path=data_p,
            output_dir=tmp_path / "ckpt",
            steps=3,
            batch_size=1,
            block_size=64,
            device="cpu",
        )

        trainer = BharatSFTTrainer(config)
        dataset = trainer.prepare_dataset()
        assert len(dataset) > 0

        # Verify assistant loss masking (-100 on prompt tokens)
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == target_ids.shape
        assert (target_ids == -100).any()  # Prompt is masked
        assert (target_ids != -100).any()  # Response has active loss targets

    def test_sft_training_tiny(self, tmp_path: Path):
        data_p = tmp_path / "sft_data.jsonl"
        export_instruction_curriculum(data_p)
        out_dir = tmp_path / "sft_out"

        config = SFTTrainingConfig(
            tier="tiny",
            data_path=data_p,
            output_dir=out_dir,
            steps=3,
            batch_size=1,
            block_size=64,
            device="cpu",
        )

        trainer = BharatSFTTrainer(config)
        res = trainer.train()

        assert res.tier == "tiny"
        assert res.final_loss > 0.0
        assert res.active_tokens > 0
        assert Path(res.checkpoint_path).is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "1b", "--steps", "25", "--learning-rate", "1e-5"])
        assert args.tier == "1b"
        assert args.steps == 25
        assert args.learning_rate == 1e-5

    def test_cli_main(self, tmp_path: Path):
        data_p = tmp_path / "cli_sft.jsonl"
        export_instruction_curriculum(data_p)
        out_dir = tmp_path / "cli_out"

        code = train_sft_main(
            [
                "--tier",
                "tiny",
                "--data-path",
                str(data_p),
                "--output-dir",
                str(out_dir),
                "--steps",
                "2",
                "--batch-size",
                "1",
                "--block-size",
                "64",
                "--device",
                "cpu",
            ]
        )
        assert code == 0
        assert (out_dir / "final.pt").is_file()
