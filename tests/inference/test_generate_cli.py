from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from inference.generate import generate, load_checkpoint
from inference.generate import main as generate_cli_main


class DummyCharTokenizer(BharatTokenizer):
    """Deterministic small tokenizer for testing generation."""

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

    def get_metadata(self) -> dict[str, object]:
        return {"vocab_size": self.vocab_size}

    def fingerprint(self) -> str:
        return "dummy_char_fp"


@pytest.fixture
def dummy_bharat_checkpoint(tmp_path: Path) -> Path:
    config = BharatModelConfig(
        vocab_size=50257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = BharatForCausalLM(config)
    ckpt_file = tmp_path / "bharat_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "step": 50,
        },
        ckpt_file,
    )
    return ckpt_file


class TestGenerateCLI:
    def test_load_bharat_checkpoint(self, dummy_bharat_checkpoint: Path) -> None:
        model, model_cfg, tokenizer = load_checkpoint(dummy_bharat_checkpoint, device="cpu")
        assert isinstance(model, BharatForCausalLM)
        assert isinstance(model_cfg, BharatModelConfig)
        assert model_cfg.vocab_size == 50257
        assert tokenizer is not None

    def test_generate_text(self, dummy_bharat_checkpoint: Path) -> None:
        model, _, _ = load_checkpoint(dummy_bharat_checkpoint, device="cpu")
        tokenizer = DummyCharTokenizer()

        out_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Hello",
            max_tokens=10,
            temperature=1.0,
            top_k=10,
            top_p=0.9,
            device="cpu",
            show_speed=False,
        )
        assert isinstance(out_text, str)
        assert len(out_text) > 0

    def test_cli_single_prompt(self, dummy_bharat_checkpoint: Path) -> None:
        ret = generate_cli_main(
            [
                "--checkpoint",
                str(dummy_bharat_checkpoint),
                "--prompt",
                "Test prompt",
                "--max-tokens",
                "5",
                "--device",
                "cpu",
            ]
        )
        assert ret == 0
