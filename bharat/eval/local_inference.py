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
    ) -> list[str]: ...


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


def _build_real_generate_fn(
    ckpt_path: str | Path,
    tok_path: str | Path,
    default_device: str,
    default_max_new_tokens: int,
) -> BatchGenerator:
    import torch

    from bharat.models.bharat_model import BharatForCausalLM
    from bharat.models.generation import generate
    from bharat.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(str(tok_path))
    ckpt_p = Path(ckpt_path)
    if ckpt_p.is_file():
        from bharat.models.config import BharatModelConfig

        ckpt = torch.load(ckpt_p, map_location=default_device, weights_only=False)
        if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
            config = BharatModelConfig.from_dict(ckpt["config"])
            model = BharatForCausalLM(config)
            model.load_state_dict(ckpt.get("model", ckpt))
        else:
            raise ValueError(
                f"Checkpoint file '{ckpt_path}' must contain a valid 'config' dictionary"
            )
    else:
        model = BharatForCausalLM.from_pretrained(str(ckpt_path), map_location=default_device)
    model.eval()

    def _generate(
        prompts: list[str],
        max_new_tokens: int | None = None,
        device: str | None = None,
    ) -> list[str]:
        batch = tokenizer.encode_batch(prompts)
        max_len = max(len(ids) for ids in batch)
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[list[int]] = []
        for ids in batch:
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask_list.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        use_device = device or default_device
        if use_device:
            input_ids = input_ids.to(use_device)
            attention_mask = attention_mask.to(use_device)

        output_ids = generate(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or default_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        results: list[str] = []
        for index, prompt in enumerate(prompts):
            new_ids = output_ids[index, max_len:]
            completion = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
            results.append(prompt + completion)
        return results

    return _generate


def load_local_causal_lm_adapter(
    config: LocalInferenceConfig,
) -> LocalCausalLMAdapter:
    generate_fn = _build_real_generate_fn(
        ckpt_path=config.checkpoint,
        tok_path=config.tokenizer,
        default_device=config.device,
        default_max_new_tokens=config.max_new_tokens,
    )
    return LocalCausalLMAdapter(config, generate_fn=generate_fn)
