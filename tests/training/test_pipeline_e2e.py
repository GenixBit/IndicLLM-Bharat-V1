from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

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
        return "dummy_fingerprint_0123456789abcdef"


@pytest.fixture
def dummy_sft_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "sft.jsonl"
    records = [
        {"instruction": "Namaste", "response": "Namaste! Kaise hain aap?"},
        {"instruction": "2+2 kitna hota hai?", "response": "2+2 4 hota hai."},
        {"instruction": "Translate hello to Hindi", "response": "Namaste"},
    ]
    with data_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return data_file


@pytest.fixture
def dummy_dpo_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "dpo.jsonl"
    records = [
        {
            "prompt": "Capital of India?",
            "chosen": "New Delhi is the capital of India.",
            "rejected": "Mumbai is the capital.",
        },
        {
            "prompt": "What is Python?",
            "chosen": "Python is a programming language.",
            "rejected": "Python is only a snake.",
        },
    ]
    with data_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return data_file


@pytest.fixture
def dummy_eval_data(tmp_path: Path) -> Path:
    eval_file = tmp_path / "eval.jsonl"
    records = [
        {
            "example_id": "ex_001",
            "task_type": "text_classification",
            "prompt": "Is this positive: Great work!",
            "reference": "positive",
            "metadata": {"task": "sentiment"},
        }
    ]
    with eval_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return eval_file


class TestPipelineE2E:
    def test_pipeline_config_from_yaml(self) -> None:
        yaml_path = Path("configs/pipeline/bharat-350m-e2e.yaml")
        assert yaml_path.is_file(), f"Recipe not found: {yaml_path}"

        config = PipelineConfig.from_yaml(yaml_path)
        assert config.name == "bharat-350m-e2e"
        assert config.pretrain.enabled is True
        assert config.sft.enabled is True
        assert config.dpo.enabled is True
        assert config.eval.enabled is True
        assert config.pretrain.learning_rate == 3.0e-4
        assert config.dpo.beta == 0.1

    def test_pipeline_cli_dry_run(self, capsys) -> None:
        yaml_path = Path("configs/pipeline/bharat-350m-e2e.yaml")
        ret = pipeline_cli_main(["--config", str(yaml_path), "--dry-run", "--json"])
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["dry_run"] is True
        assert data["pipeline_name"] == "bharat-350m-e2e"
        assert data["stages"]["pretrain"] is True
        assert data["stages"]["sft"] is True
        assert data["stages"]["dpo"] is True

    def test_run_pipeline_e2e_synthetic(
        self,
        tmp_path: Path,
        dummy_sft_data: Path,
        dummy_dpo_data: Path,
        dummy_eval_data: Path,
    ) -> None:
        """Runs the full lifecycle: pretraining -> SFT -> DPO -> Eval on synthetic/dummy data."""
        torch.manual_seed(42)
        out_dir = tmp_path / "pipeline_run"
        tokenizer = DummyCharTokenizer()

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
                block_size=64,
                learning_rate=5e-4,
                warmup_iters=1,
                device="cpu",
            ),
            dpo=DPOStageConfig(
                enabled=True,
                data_path=str(dummy_dpo_data),
                max_iters=3,
                batch_size=1,
                block_size=64,
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
