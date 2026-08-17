from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bharat.eval.local_inference import LocalInferenceConfig, load_local_causal_lm_adapter
from bharat.eval.schema import EvalExample
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.training.checkpointing import save_checkpoint


def make_tiny_config(vocab_size: int = 512) -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        initializer_range=0.02,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )


class TestLocalInferenceCheckpointLoading:
    def test_load_from_pt_file(
        self, tmp_path: Path, tiny_tokenizer, tiny_bpe_tokenizer_json: str
    ) -> None:
        """Verify that LocalCausalLMAdapter can load and run generation from a single-file .pt checkpoint."""
        torch.manual_seed(42)
        cfg = make_tiny_config(vocab_size=tiny_tokenizer.vocab_size)
        model = BharatForCausalLM(cfg)

        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            path=ckpt_path,
            model=model,
            config=cfg.to_dict(),
            step=10,
            seed=42,
        )

        inf_cfg = LocalInferenceConfig(
            checkpoint=ckpt_path,
            tokenizer=tiny_bpe_tokenizer_json,
            max_new_tokens=8,
            device="cpu",
        )
        adapter = load_local_causal_lm_adapter(inf_cfg)

        example = EvalExample(
            example_id="ex-1",
            task_type="qa",
            prompt="Hello world",
            expected="Hello",
        )

        prediction = adapter.predict(example)
        assert isinstance(prediction, str)

    def test_load_from_directory(
        self, tmp_path: Path, tiny_tokenizer, tiny_bpe_tokenizer_json: str
    ) -> None:
        """Verify that LocalCausalLMAdapter can load and run generation from a save_pretrained directory."""
        torch.manual_seed(42)
        cfg = make_tiny_config(vocab_size=tiny_tokenizer.vocab_size)
        model = BharatForCausalLM(cfg)

        model_dir = tmp_path / "saved_model"
        model.save_pretrained(str(model_dir))

        inf_cfg = LocalInferenceConfig(
            checkpoint=model_dir,
            tokenizer=tiny_bpe_tokenizer_json,
            max_new_tokens=8,
            device="cpu",
        )
        adapter = load_local_causal_lm_adapter(inf_cfg)

        example = EvalExample(
            example_id="ex-2",
            task_type="qa",
            prompt="What is AI",
            expected="AI is intelligence",
        )

        prediction = adapter.predict(example)
        assert isinstance(prediction, str)

    def test_invalid_pt_file_without_config_raises(
        self, tmp_path: Path, tiny_bpe_tokenizer_json: str
    ) -> None:
        """Verify that a raw .pt file without config metadata raises ValueError with clear message."""
        bad_ckpt = tmp_path / "bad.pt"
        torch.save({"raw_tensor": torch.randn(10)}, bad_ckpt)

        inf_cfg = LocalInferenceConfig(
            checkpoint=bad_ckpt,
            tokenizer=tiny_bpe_tokenizer_json,
            max_new_tokens=8,
        )

        with pytest.raises(ValueError, match="must contain a valid 'config' dictionary"):
            load_local_causal_lm_adapter(inf_cfg)
