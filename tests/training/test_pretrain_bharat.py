from __future__ import annotations

from pathlib import Path

from bharat.data.mixture import stream_and_pack_mixture
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import load_tokenizer
from train.pretrain_bharat import get_cosine_lr, parse_args
from train.pretrain_bharat import main as pretrain_main


class TestPretrainBharat:
    def test_cosine_lr_schedule(self):
        # Warmup phase
        lr_0 = get_cosine_lr(0, max_steps=100, warmup_steps=10, base_lr=1e-3)
        assert lr_0 == 1e-4

        lr_9 = get_cosine_lr(9, max_steps=100, warmup_steps=10, base_lr=1e-3)
        assert abs(lr_9 - 1e-3) < 1e-6

        # Decay phase
        lr_50 = get_cosine_lr(50, max_steps=100, warmup_steps=10, base_lr=1e-3)
        assert 1e-4 < lr_50 < 1e-3

        # Post-max steps
        lr_120 = get_cosine_lr(120, max_steps=100, warmup_steps=10, base_lr=1e-3)
        assert lr_120 == 1e-4

    def test_pretrain_bharat_single_step(self, tmp_path: Path):
        shards_dir = tmp_path / "shards"
        tok = load_tokenizer("gpt2")
        stream_and_pack_mixture(tok, shards_dir, max_tokens_per_shard=1000, max_docs=5)

        cfg_file = tmp_path / "tiny_config.yaml"
        cfg = BharatModelConfig(
            vocab_size=tok.vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        import yaml

        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f)

        out_dir = tmp_path / "checkpoints"

        code = pretrain_main(
            [
                "--config",
                str(cfg_file),
                "--shards-dir",
                str(shards_dir),
                "--output-dir",
                str(out_dir),
                "--max-steps",
                "3",
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

    def test_cli_parse_args(self):
        args = parse_args(["--config", "custom.yaml", "--max-steps", "50"])
        assert args.config == "custom.yaml"
        assert args.max_steps == 50
