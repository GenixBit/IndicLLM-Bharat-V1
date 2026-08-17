from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from bharat.training.pipeline import (
    DPOStageConfig,
    EvalStageConfig,
    PipelineConfig,
    PretrainStageConfig,
    SFTStageConfig,
    run_pipeline,
)
from scripts.run_pipeline import main as pipeline_cli_main


class DummyCharTokenizer(BharatTokenizer):
    """Deterministic small tokenizer for fast pipeline unit tests."""

    def __init__(self) -> None:
        self.vocab = {chr(i): i for i in range(128)}
        self.inv_vocab = {i: chr(i) for i in range(128)}

    @property
    def vocab_size(self) -> int:
        return 128

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 1

    @property
    def tokenizer_type(self) -> str:
        return "dummy_char"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab.get(c, 2) for c in text]

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.inv_vocab.get(t, "?") for t in ids)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def get_metadata(self) -> dict[str, Any]:
        return {"vocab_size": self.vocab_size, "type": "dummy_char"}

    def fingerprint(self) -> str:
        return "dummy_char_fingerprint"


@pytest.fixture
def dummy_sft_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "dummy_sft.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4."},
                ]
            }
        ),
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Capital of India?"},
                    {"role": "assistant", "content": "New Delhi."},
                ]
            }
        ),
    ]
    data_file.write_text("\n".join(lines), encoding="utf-8")
    return data_file


@pytest.fixture
def dummy_dpo_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "dummy_dpo.jsonl"
    lines = [
        json.dumps(
            {
                "prompt": "Explain gravity in one line.",
                "chosen": "Gravity is the attractive force between masses.",
                "rejected": "Gravity is magic that pulls things down.",
            }
        ),
        json.dumps(
            {
                "prompt": "Namaste meaning?",
                "chosen": "A respectful greeting in India.",
                "rejected": "A random word.",
            }
        ),
    ]
    data_file.write_text("\n".join(lines), encoding="utf-8")
    return data_file


class TestPipelineOrchestrator:
    def test_pipeline_config_serialization(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(
            name="test-pipeline",
            output_dir=str(tmp_path / "output"),
            tokenizer_path="data/indic/tokenizer.json",
            pretrain=PretrainStageConfig(
                enabled=True,
                max_iters=10,
                device="cpu",
            ),
            sft=SFTStageConfig(
                enabled=True,
                data_path=str(tmp_path / "sft.jsonl"),
                max_iters=5,
                device="cpu",
            ),
            dpo=DPOStageConfig(
                enabled=True,
                data_path=str(tmp_path / "dpo.jsonl"),
                max_iters=5,
                device="cpu",
            ),
            eval=EvalStageConfig(
                enabled=False,
            ),
        )

        yaml_path = tmp_path / "pipeline.yaml"
        cfg.to_yaml(yaml_path)

        loaded = PipelineConfig.from_yaml(yaml_path)
        assert loaded.name == "test-pipeline"
        assert loaded.pretrain.max_iters == 10
        assert loaded.sft.max_iters == 5
        assert loaded.dpo.max_iters == 5
        assert not loaded.eval.enabled

    def test_pipeline_cli_dry_run(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_content = """
name: "test-cli-pipeline"
output_dir: "output/test"
tokenizer_path: "data/indic/tokenizer.json"
pretrain:
  enabled: true
  max_iters: 10
  synthetic_data: true
sft:
  enabled: false
dpo:
  enabled: false
eval:
  enabled: false
"""
        yaml_path.write_text(yaml_content, encoding="utf-8")

        ret = pipeline_cli_main(["--config", str(yaml_path), "--dry-run", "--json"])
        assert ret == 0

    def test_run_pipeline_end_to_end_fast(
        self,
        tmp_path: Path,
        dummy_sft_data: Path,
        dummy_dpo_data: Path,
    ) -> None:
        torch.manual_seed(42)
        tokenizer = DummyCharTokenizer()
        out_dir = tmp_path / "pipeline_run"

        config = PipelineConfig(
            name="test-mini-e2e",
            output_dir=str(out_dir),
            tokenizer_path="",
            pretrain=PretrainStageConfig(
                enabled=True,
                synthetic_data=True,
                max_iters=5,
                batch_size=2,
                seq_len=32,
                learning_rate=1e-3,
                warmup_iters=2,
                device="cpu",
                dtype="float32",
            ),
            sft=SFTStageConfig(
                enabled=True,
                data_path=str(dummy_sft_data),
                max_iters=3,
                batch_size=1,
                block_size=512,
                learning_rate=1e-4,
                device="cpu",
            ),
            dpo=DPOStageConfig(
                enabled=True,
                data_path=str(dummy_dpo_data),
                max_iters=3,
                batch_size=1,
                block_size=512,
                learning_rate=1e-4,
                beta=0.1,
                device="cpu",
            ),
            eval=EvalStageConfig(
                enabled=False,  # Skip eval stage for DummyCharTokenizer (no local file on disk)
            ),
            seed=42,
        )

        result = run_pipeline(config, tokenizer=tokenizer)

        assert result.pipeline_name == "test-mini-e2e"
        assert "pretrain" in result.completed_stages
        assert "sft" in result.completed_stages
        assert "dpo" in result.completed_stages

        assert result.pretrain_checkpoint is not None
        assert Path(result.pretrain_checkpoint).is_file()
        assert result.sft_checkpoint is not None
        assert Path(result.sft_checkpoint).is_file()
        assert result.dpo_checkpoint is not None
        assert Path(result.dpo_checkpoint).is_file()

        assert result.pretrain_loss is not None and result.pretrain_loss > 0
        assert result.sft_loss is not None and result.sft_loss > 0
        assert result.dpo_loss is not None
        assert result.total_duration_sec > 0

        summary_file = out_dir / "pipeline_summary.json"
        assert summary_file.is_file()
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary["pipeline_name"] == "test-mini-e2e"

    @pytest.mark.parametrize(
        "tier",
        ["350m", "1b", "3b", "7b"],
    )
    def test_pipeline_recipes_load_valid(self, tier: str) -> None:
        recipe_path = Path(f"configs/pipeline/bharat-{tier}-e2e.yaml")
        assert recipe_path.is_file()

        cfg = PipelineConfig.from_yaml(recipe_path)
        assert cfg.name == f"bharat-{tier}-e2e"
        assert cfg.pretrain.model_config_path is not None
        assert Path(cfg.pretrain.model_config_path).is_file()

        model_cfg = BharatModelConfig.from_yaml(cfg.pretrain.model_config_path)
        assert model_cfg.vocab_size == 64000
        assert model_cfg.num_hidden_layers > 0
        assert model_cfg.hidden_size > 0
