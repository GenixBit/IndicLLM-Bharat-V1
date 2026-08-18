from __future__ import annotations

from pathlib import Path

import yaml

from bharat.models.bharat_model import BharatDecoderLayer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DISTRIBUTED_CONFIGS_DIR = ROOT_DIR / "configs" / "distributed"


class TestDistributedConfigs:
    def test_distributed_configs_directory_exists(self):
        assert DISTRIBUTED_CONFIGS_DIR.is_dir()

    def test_accelerate_ddp_valid(self):
        p = DISTRIBUTED_CONFIGS_DIR / "accelerate_ddp.yaml"
        assert p.is_file()
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["distributed_type"] == "MULTI_GPU"
        assert cfg["mixed_precision"] in ("bf16", "fp16")
        assert cfg["num_processes"] >= 1

    def test_accelerate_fsdp_valid(self):
        p = DISTRIBUTED_CONFIGS_DIR / "accelerate_fsdp.yaml"
        assert p.is_file()
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["distributed_type"] == "FSDP"
        assert cfg["mixed_precision"] in ("bf16", "fp16")
        fsdp_cfg = cfg["fsdp_config"]
        assert fsdp_cfg["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
        assert fsdp_cfg["fsdp_transformer_layer_cls_to_wrap"] == BharatDecoderLayer.__name__
        assert fsdp_cfg["fsdp_sharding_strategy"] in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD")

    def test_deepspeed_zero2_valid(self):
        p = DISTRIBUTED_CONFIGS_DIR / "deepspeed_zero2.yaml"
        assert p.is_file()
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["zero_optimization"]["stage"] == 2
        assert cfg["bf16"]["enabled"] is True

    def test_deepspeed_zero3_valid(self):
        p = DISTRIBUTED_CONFIGS_DIR / "deepspeed_zero3.yaml"
        assert p.is_file()
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["zero_optimization"]["stage"] == 3
        assert "activation_checkpointing" in cfg
        assert cfg["bf16"]["enabled"] is True
