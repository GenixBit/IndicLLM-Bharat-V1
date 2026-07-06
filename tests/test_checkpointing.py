from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.checkpointing import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_for_legacy,
)


@pytest.fixture
def simple_model():
    return nn.Linear(10, 10)


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def gpt2_tokenizer():
    return load_tokenizer("gpt2")


class TestCheckpointMetadata:
    def test_default_metadata(self):
        meta = CheckpointMetadata()
        assert meta.tokenizer_type == ""
        assert meta.tokenizer_hash == ""
        assert meta.vocab_size == 0

    def test_metadata_roundtrip(self):
        meta = CheckpointMetadata(
            tokenizer_type="gpt2",
            tokenizer_hash="abc123",
            vocab_size=50257,
            git_sha="deadbeef",
            data_version="v1.0",
            seed=42,
            training_step=1000,
        )
        import dataclasses
        d = dataclasses.asdict(meta)
        restored = CheckpointMetadata(**d)
        assert restored.tokenizer_type == "gpt2"
        assert restored.tokenizer_hash == "abc123"
        assert restored.vocab_size == 50257
        assert restored.git_sha == "deadbeef"
        assert restored.data_version == "v1.0"
        assert restored.seed == 42
        assert restored.training_step == 1000


class TestCheckpointSaveLoad:
    def test_save_load_basic(self, tmp_path, simple_model):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, simple_model)
        assert path.exists()

        loaded_model = nn.Linear(10, 10)
        result = load_checkpoint(path, loaded_model, device="cpu", strict=False)
        assert "metadata" in result
        assert isinstance(result["metadata"], CheckpointMetadata)

    def test_save_with_optimizer(self, tmp_path, simple_model, simple_optimizer):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, simple_model, optimizer=simple_optimizer)
        loaded_model = nn.Linear(10, 10)
        loaded_opt = torch.optim.SGD(loaded_model.parameters(), lr=0.01)
        result = load_checkpoint(path, loaded_model, optimizer=loaded_opt, device="cpu", strict=False)
        assert result["metadata"].training_step == 0

    def test_save_with_tokenizer(self, tmp_path, simple_model, gpt2_tokenizer):
        path = tmp_path / "ckpt_with_tok.pt"
        save_checkpoint(path, simple_model, tokenizer=gpt2_tokenizer)
        loaded_model = nn.Linear(10, 10)
        result = load_checkpoint(path, loaded_model, tokenizer=gpt2_tokenizer, device="cpu", strict=False)
        meta = result["metadata"]
        assert meta.tokenizer_type == "gpt2"
        assert meta.vocab_size == 50257
        assert len(meta.tokenizer_hash) == 64

    def test_load_wrong_tokenizer_fails(self, tmp_path, simple_model, gpt2_tokenizer):
        path = tmp_path / "ckpt_wrong.pt"
        save_checkpoint(path, simple_model, tokenizer=gpt2_tokenizer)

        class FakeTokenizer(BharatTokenizer):
            @property
            def vocab_size(self):
                return 100
            @property
            def eos_token_id(self):
                return 0
            @property
            def pad_token_id(self):
                return 0
            @property
            def tokenizer_type(self):
                return "fake"
            def encode(self, text, add_special_tokens=True):
                return []
            def encode_batch(self, texts, add_special_tokens=True):
                return [[] for _ in texts]
            def decode(self, ids, skip_special_tokens=True):
                return ""
            def decode_batch(self, batch, skip_special_tokens=True):
                return ["" for _ in batch]
            def get_metadata(self):
                return {}

        loaded_model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match=r"Tokenizer mismatch|Vocab size mismatch"):
            load_checkpoint(path, loaded_model, tokenizer=FakeTokenizer(), device="cpu", strict=False)

    def test_missing_metadata_fails(self, tmp_path, simple_model):
        path = tmp_path / "ckpt_no_meta.pt"
        torch.save({"model": simple_model.state_dict()}, path)
        loaded_model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="no metadata"):
            load_checkpoint(path, loaded_model, device="cpu", strict=False)

    def test_nonexistent_path(self, simple_model):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt", simple_model, device="cpu", strict=False)

    def test_missing_file(self, tmp_path, simple_model):
        path = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError):
            load_checkpoint(path, simple_model, device="cpu", strict=False)


class TestCheckpointMetadataFields:
    def test_metadata_has_all_fields(self, tmp_path, simple_model, gpt2_tokenizer):
        path = tmp_path / "meta_test.pt"
        save_checkpoint(
            path,
            simple_model,
            tokenizer=gpt2_tokenizer,
            step=500,
            seed=12345,
            data_version="v2.0",
        )
        loaded_model = nn.Linear(10, 10)
        result = load_checkpoint(path, loaded_model, tokenizer=gpt2_tokenizer, device="cpu", strict=False)
        meta = result["metadata"]
        assert meta.training_step == 500
        assert meta.seed == 12345
        assert meta.data_version == "v2.0"
        assert meta.torch_version.startswith("2.") or meta.torch_version.startswith("1.")

    def test_git_sha_present(self, tmp_path, simple_model):
        path = tmp_path / "git_test.pt"
        save_checkpoint(path, simple_model)
        loaded_model = nn.Linear(10, 10)
        result = load_checkpoint(path, loaded_model, device="cpu", strict=False)
        sha = result["metadata"].git_sha
        assert len(sha) == 40, f"Expected 40-char SHA, got '{sha}' (len={len(sha)})"


class TestCheckpointMismatch:
    def test_vocab_mismatch(self, tmp_path, simple_model, gpt2_tokenizer):
        path = tmp_path / "vocab_mismatch.pt"
        save_checkpoint(path, simple_model, tokenizer=gpt2_tokenizer)

        class DifferentVocabTokenizer(BharatTokenizer):
            @property
            def vocab_size(self):
                return 999
            @property
            def eos_token_id(self):
                return 0
            @property
            def pad_token_id(self):
                return 0
            @property
            def tokenizer_type(self):
                return "gpt2"
            def encode(self, text, add_special_tokens=True):
                return []
            def encode_batch(self, texts, add_special_tokens=True):
                return [[] for _ in texts]
            def decode(self, ids, skip_special_tokens=True):
                return ""
            def decode_batch(self, batch, skip_special_tokens=True):
                return ["" for _ in batch]
            def get_metadata(self):
                return {}

        loaded_model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match=r"Tokenizer mismatch|Vocab size mismatch"):
            load_checkpoint(path, loaded_model, tokenizer=DifferentVocabTokenizer(), device="cpu", strict=False)


class TestLegacyCheckpoint:
    def test_legacy_save_format(self, tmp_path):
        path = tmp_path / "legacy.pt"
        state = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        save_checkpoint_for_legacy(path, state)
        assert path.exists()

        ckpt = torch.load(path, weights_only=False)
        assert "model" in ckpt
        assert "metadata" in ckpt
        assert "step" in ckpt

    def test_legacy_with_tokenizer(self, tmp_path, gpt2_tokenizer):
        path = tmp_path / "legacy_tok.pt"
        state = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        save_checkpoint_for_legacy(path, state, tokenizer=gpt2_tokenizer)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["metadata"]["tokenizer_type"] == "gpt2"
        assert ckpt["metadata"]["vocab_size"] == 50257
