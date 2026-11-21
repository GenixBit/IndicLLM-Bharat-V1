from __future__ import annotations

import pytest
import torch

from bharat.inference import optimized_engine
from bharat.inference.optimized_engine import (
    OptimizedInferenceEngine,
    _load_checked_weights,
)


class _TinyTokenizer:
    vocab_size = 128

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) % self.vocab_size for char in text] or [1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(token) % 95 + 32) for token in ids)


def _patch_tiny_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimized_engine, "load_tokenizer", lambda _source: _TinyTokenizer())


def test_non_demo_tier_requires_checkpoint_before_model_allocation() -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint is required"):
        OptimizedInferenceEngine(tier="1b", device="cpu")


def test_demo_without_checkpoint_is_explicitly_untrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_tokenizer(monkeypatch)
    engine = OptimizedInferenceEngine(tier="tiny", device="cpu", max_seq_len=32)
    assert engine.checkpoint_loaded is False


@pytest.mark.parametrize("envelope", ["raw", "model_state_dict", "state_dict"])
def test_complete_local_checkpoint_loads_strictly(
    tmp_path, monkeypatch: pytest.MonkeyPatch, envelope: str
) -> None:
    _patch_tiny_tokenizer(monkeypatch)
    original = OptimizedInferenceEngine(tier="tiny", device="cpu", max_seq_len=32)
    weights = {name: value.detach().clone() for name, value in original.model.state_dict().items()}
    payload = weights if envelope == "raw" else {envelope: weights}
    checkpoint = tmp_path / "tiny.pt"
    torch.save(payload, checkpoint)

    restored = OptimizedInferenceEngine(
        tier="tiny", checkpoint_path=checkpoint, device="cpu", max_seq_len=32
    )
    assert restored.checkpoint_loaded is True
    for name, expected in weights.items():
        assert torch.equal(restored.model.state_dict()[name], expected)


def test_missing_supplied_checkpoint_never_falls_back_to_random_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tiny_tokenizer(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
        OptimizedInferenceEngine(
            tier="tiny", checkpoint_path=tmp_path / "missing.pt", device="cpu"
        )


def test_incompatible_partial_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(2, 3)}}, path)
    with pytest.raises(ValueError, match="incompatible"):
        _load_checked_weights(torch.nn.Linear(3, 2), path)


def test_non_tensor_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "non-tensor.pt"
    torch.save({"state_dict": {"weight": 12}}, path)
    with pytest.raises(ValueError, match="nonempty tensor weights"):
        _load_checked_weights(torch.nn.Linear(3, 2), path)


def test_invalid_checkpoint_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"not a pytorch checkpoint")
    with pytest.raises(ValueError, match="Unable to safely decode"):
        _load_checked_weights(torch.nn.Linear(3, 2), path)
