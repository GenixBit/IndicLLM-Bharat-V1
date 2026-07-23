from __future__ import annotations

from pathlib import Path

import pytest

from bharat.eval.local_inference import (
    LocalCausalLMAdapter,
    LocalInferenceConfig,
    load_local_causal_lm_adapter,
)
from bharat.eval.schema import EvalExample


def _make_dirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestLocalInferenceConfig:
    def test_remote_checkpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="Remote checkpoint path rejected"):
            LocalInferenceConfig(
                checkpoint="https://example.com/model",
                tokenizer="/fake/tokenizer",
            )

    def test_remote_tokenizer_rejected(self) -> None:
        with pytest.raises(ValueError, match="Remote tokenizer path rejected"):
            LocalInferenceConfig(
                checkpoint="/fake/checkpoint",
                tokenizer="s3://bucket/tokenizer",
            )

    def test_negative_max_new_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            LocalInferenceConfig(
                checkpoint="/fake/checkpoint",
                tokenizer="/fake/tokenizer",
                max_new_tokens=-1,
            )

    def test_zero_max_new_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            LocalInferenceConfig(
                checkpoint="/fake/checkpoint",
                tokenizer="/fake/tokenizer",
                max_new_tokens=0,
            )

    def test_valid_local_paths(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(
            checkpoint=str(ckpt),
            tokenizer=str(tok),
            device="cuda",
            max_new_tokens=512,
        )
        assert str(config.checkpoint) == str(ckpt)
        assert str(config.tokenizer) == str(tok)
        assert config.device == "cuda"
        assert config.max_new_tokens == 512


class TestLocalCausalLMAdapter:
    def test_missing_checkpoint_rejected(self, tmp_path: Path) -> None:
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(
            checkpoint=tmp_path / "nonexistent_ckpt",
            tokenizer=tok,
        )
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            LocalCausalLMAdapter(config)

    def test_missing_tokenizer_rejected(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        config = LocalInferenceConfig(
            checkpoint=ckpt,
            tokenizer=tmp_path / "nonexistent_tok",
        )
        with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
            LocalCausalLMAdapter(config)

    def test_empty_prompt_rejected(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(checkpoint=ckpt, tokenizer=tok)

        def fake_generate(
            prompts: list[str],
            max_new_tokens: int | None = None,
            device: str | None = None,
        ) -> list[str]:
            return [""]

        adapter = LocalCausalLMAdapter(config, generate_fn=fake_generate)
        ex = EvalExample(
            example_id="test_001",
            task_type="qa",
            prompt="   ",
            expected="",
        )
        with pytest.raises(ValueError, match="Empty prompt cannot be tokenized"):
            adapter.predict(ex)

    def test_decodes_only_new_tokens(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(checkpoint=ckpt, tokenizer=tok)

        def fake_generate(
            prompts: list[str],
            max_new_tokens: int | None = None,
            device: str | None = None,
        ) -> list[str]:
            return [prompts[0] + " world"]

        adapter = LocalCausalLMAdapter(config, generate_fn=fake_generate)
        ex = EvalExample(
            example_id="test_001",
            task_type="qa",
            prompt="Hello",
            expected="",
        )
        result = adapter.predict(ex)
        assert result == " world"

    def test_decodes_only_new_tokens_long_prompt(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(checkpoint=ckpt, tokenizer=tok)

        def fake_generate(
            prompts: list[str],
            max_new_tokens: int | None = None,
            device: str | None = None,
        ) -> list[str]:
            return [prompts[0] + " how are you today"]

        adapter = LocalCausalLMAdapter(config, generate_fn=fake_generate)
        ex = EvalExample(
            example_id="test_001",
            task_type="qa",
            prompt="Hello there,",
            expected="",
        )
        result = adapter.predict(ex)
        assert result == " how are you today"

    def test_injected_generate_fn_receives_config_params(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(
            checkpoint=ckpt,
            tokenizer=tok,
            device="xla",
            max_new_tokens=42,
        )

        captured: dict[str, object] = {}

        def fake_generate(
            prompts: list[str],
            max_new_tokens: int | None = None,
            device: str | None = None,
        ) -> list[str]:
            captured["max_new_tokens"] = max_new_tokens
            captured["device"] = device
            return [prompts[0] + " out"]

        adapter = LocalCausalLMAdapter(config, generate_fn=fake_generate)
        ex = EvalExample(
            example_id="test_001",
            task_type="qa",
            prompt="in",
            expected="",
        )
        adapter.predict(ex)
        assert captured["max_new_tokens"] == 42
        assert captured["device"] == "xla"


class TestLoadLocalCausalLMAdapter:
    def test_returns_adapter_with_valid_config(self, tmp_path: Path) -> None:
        ckpt = _make_dirs(tmp_path / "checkpoint")
        tok = _make_dirs(tmp_path / "tokenizer")
        config = LocalInferenceConfig(checkpoint=ckpt, tokenizer=tok)
        adapter = load_local_causal_lm_adapter(config)
        assert isinstance(adapter, LocalCausalLMAdapter)

    def test_rejects_remote_checkpoint(self) -> None:
        with pytest.raises(ValueError, match="Remote checkpoint path rejected"):
            load_local_causal_lm_adapter(
                LocalInferenceConfig(
                    checkpoint="https://example.com/model",
                    tokenizer="/fake/tok",
                )
            )

    def test_rejects_remote_tokenizer(self) -> None:
        with pytest.raises(ValueError, match="Remote tokenizer path rejected"):
            load_local_causal_lm_adapter(
                LocalInferenceConfig(
                    checkpoint="/fake/ckpt",
                    tokenizer="gs://bucket/tokenizer",
                )
            )
