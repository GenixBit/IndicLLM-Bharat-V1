from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bharat.eval.schema import EvalExample

_URL_RE = re.compile(r"^(https?|ftp|s3|gs):/+", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


class BatchGenerator(Protocol):
    def __call__(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
        device: str | None = None,
    ) -> list[str]:
        ...


@dataclass(frozen=True)
class LocalInferenceConfig:
    checkpoint: str | Path
    tokenizer: str | Path
    device: str = "cpu"
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        paths = (
            ("checkpoint", self.checkpoint),
            ("tokenizer", self.tokenizer),
        )
        for name, value in paths:
            path_str = str(value)
            if _is_remote_url(path_str):
                raise ValueError(f"Remote {name} path rejected: {path_str}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


class LocalCausalLMAdapter:
    def __init__(
        self,
        config: LocalInferenceConfig,
        generate_fn: BatchGenerator | None = None,
    ) -> None:
        self._config = config
        checkpoint_path = Path(config.checkpoint)
        tokenizer_path = Path(config.tokenizer)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        self._generate_fn = generate_fn or _default_generate

    def predict(self, example: EvalExample) -> str:
        if not example.prompt.strip():
            raise ValueError("Empty prompt cannot be tokenized")
        full_texts = self._generate_fn(
            [example.prompt],
            max_new_tokens=self._config.max_new_tokens,
            device=self._config.device,
        )
        full_text = full_texts[0]
        return full_text[len(example.prompt) :]


def _default_generate(
    prompts: list[str],
    max_new_tokens: int | None = None,
    device: str | None = None,
) -> list[str]:
    del prompts, max_new_tokens, device
    raise NotImplementedError(
        "Real model generation requires a loaded model and tokenizer. "
        "Inject a generate_fn or use load_local_causal_lm_adapter() "
        "after model weights are available."
    )


def load_local_causal_lm_adapter(
    config: LocalInferenceConfig,
) -> LocalCausalLMAdapter:
    return LocalCausalLMAdapter(config)
