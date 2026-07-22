from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import torch

from bharat.eval.schema import EvalExample
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.generation import generate
from bharat.tokenizer import BharatTokenizer, load_tokenizer

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def _local_path(path: str | Path, *, field_name: str) -> Path:
    raw_path = str(path)
    if _is_remote_url(raw_path):
        raise ValueError(f"Remote {field_name} path rejected: {raw_path}")
    return Path(path)


class TokenGenerator(Protocol):
    def __call__(
        self,
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
        ...


@dataclass(frozen=True)
class LocalInferenceConfig:
    checkpoint_path: str | Path
    tokenizer_path: str | Path
    max_new_tokens: int = 32
    device: str = "cpu"
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        checkpoint_path = _local_path(self.checkpoint_path, field_name="checkpoint")
        tokenizer_path = _local_path(self.tokenizer_path, field_name="tokenizer")
        object.__setattr__(self, "checkpoint_path", checkpoint_path)
        object.__setattr__(self, "tokenizer_path", tokenizer_path)

        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError(
                "max_new_tokens must be an integer, "
                f"got {type(self.max_new_tokens).__name__}"
            )
        if self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {self.max_new_tokens}")
        if not self.device:
            raise ValueError("device must be a non-empty string")
        if self.do_sample and self.temperature <= 0.0:
            raise ValueError("temperature must be positive when sampling")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise TypeError(f"top_k must be an integer, got {type(self.top_k).__name__}")
            if self.top_k < 1:
                raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


def _generate_with_bharat_model(
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
    if not isinstance(model, BharatForCausalLM):
        raise TypeError(f"model must be BharatForCausalLM, got {type(model).__name__}")
    return generate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def _default_token_generator() -> TokenGenerator:
    return _generate_with_bharat_model


@dataclass(frozen=True)
class LocalCausalLMAdapter:
    model: object
    tokenizer: BharatTokenizer
    config: LocalInferenceConfig
    generator: TokenGenerator = field(default_factory=_default_token_generator)

    def predict(self, example: EvalExample) -> str:
        prompt_ids = self.tokenizer.encode(
            example.prompt,
            add_special_tokens=self.config.add_special_tokens,
        )
        if not prompt_ids:
            raise ValueError(f"Tokenizer produced no prompt IDs for {example.example_id!r}")

        device = torch.device(cast(str, self.config.device))
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            generated = self.generator(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if generated.dim() != 2 or generated.shape[0] != 1:
            raise ValueError(
                f"Generated IDs must have shape (1, sequence), got {tuple(generated.shape)}"
            )
        generated_ids = generated[0].detach().cpu().tolist()
        completion_ids = generated_ids[len(prompt_ids) :]
        return self.tokenizer.decode(
            _as_int_sequence(completion_ids),
            skip_special_tokens=True,
        ).strip()


def _as_int_sequence(values: Sequence[object]) -> list[int]:
    ids: list[int] = []
    for value in values:
        if not isinstance(value, int):
            raise TypeError(f"generated token IDs must be integers, got {type(value).__name__}")
        ids.append(value)
    return ids


def load_local_causal_lm_adapter(config: LocalInferenceConfig) -> LocalCausalLMAdapter:
    checkpoint_path = cast(Path, config.checkpoint_path)
    tokenizer_path = cast(Path, config.tokenizer_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    model = BharatForCausalLM.from_pretrained(str(checkpoint_path), map_location=config.device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    return LocalCausalLMAdapter(model=model, tokenizer=tokenizer, config=config)
