from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from bharat.eval.local_inference import (
    LocalCausalLMAdapter,
    LocalInferenceConfig,
    load_local_causal_lm_adapter,
)
from bharat.eval.schema import EvalExample
from bharat.tokenizer import BharatTokenizer


class FakeTokenizer(BharatTokenizer):
    @property
    def vocab_size(self) -> int:
        return 8

    @property
    def eos_token_id(self) -> int:
        return 7

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def tokenizer_type(self) -> str:
        return "fake"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be a boolean")
        assert text
        return [1]

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if not isinstance(skip_special_tokens, bool):
            raise TypeError("skip_special_tokens must be a boolean")
        return " ".join(f"tok_{token_id}" for token_id in ids)

    def decode_batch(
        self,
        batch: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
        }

    def fingerprint(self) -> str:
        return "fake-tokenizer"


class EmptyTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be a boolean")
        assert isinstance(text, str)
        return []


def fake_generator(
    *,
    model: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    assert model is not None
    assert attention_mask is not None
    assert max_new_tokens == 2
    assert not do_sample
    assert temperature == 1.0
    assert top_k is None
    assert top_p is None
    assert eos_token_id == 7
    assert pad_token_id == 0
    completion = torch.tensor([[2, 3]], dtype=torch.long, device=input_ids.device)
    return torch.cat([input_ids, completion], dim=-1)


def test_local_inference_config_rejects_remote_checkpoint() -> None:
    with pytest.raises(ValueError, match="Remote checkpoint path rejected"):
        LocalInferenceConfig(
            checkpoint_path="https://example.com/checkpoint",
            tokenizer_path="tokenizer.json",
        )


def test_local_inference_config_rejects_bad_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
        LocalInferenceConfig(
            checkpoint_path="checkpoint",
            tokenizer_path="tokenizer.json",
            max_new_tokens=-1,
        )


def test_local_causal_lm_adapter_decodes_generated_completion_only(
    tmp_path: Path,
) -> None:
    config = LocalInferenceConfig(
        checkpoint_path=tmp_path / "checkpoint",
        tokenizer_path=tmp_path / "tokenizer.json",
        max_new_tokens=2,
    )
    adapter = LocalCausalLMAdapter(
        model=object(),
        tokenizer=FakeTokenizer(),
        config=config,
        generator=fake_generator,
    )

    prediction = adapter.predict(
        EvalExample(
            example_id="qa_001",
            task_type="qa",
            prompt="Question?",
            expected="Answer",
        )
    )

    assert prediction == "tok_2 tok_3"


def test_local_causal_lm_adapter_rejects_empty_tokenization(
    tmp_path: Path,
) -> None:
    config = LocalInferenceConfig(
        checkpoint_path=tmp_path / "checkpoint",
        tokenizer_path=tmp_path / "tokenizer.json",
    )
    adapter = LocalCausalLMAdapter(
        model=object(),
        tokenizer=EmptyTokenizer(),
        config=config,
        generator=fake_generator,
    )

    with pytest.raises(ValueError, match="produced no prompt IDs"):
        adapter.predict(
            EvalExample(
                example_id="qa_001",
                task_type="qa",
                prompt="Question?",
                expected="Answer",
            )
        )


def test_load_local_causal_lm_adapter_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    config = LocalInferenceConfig(
        checkpoint_path=tmp_path / "missing-checkpoint",
        tokenizer_path=tokenizer_path,
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        load_local_causal_lm_adapter(config)
