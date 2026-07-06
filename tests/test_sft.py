from __future__ import annotations

import pytest


class TestSFTDataset:
    def test_dataset_initialization(self) -> None:
        pytest.skip("Needs tokenizer fixture")

    def test_padding(self) -> None:
        pytest.skip("Needs tokenizer fixture")

    def test_block_size_truncation(self) -> None:
        pytest.skip("Needs tokenizer fixture")


class TestSFTLossMasking:
    def test_assistant_only_loss(self) -> None:
        pytest.skip("PR 3: Implement after unified tokenizer")

    def test_user_tokens_masked(self) -> None:
        pytest.skip("PR 3: Implement after unified tokenizer")

    def test_system_tokens_masked(self) -> None:
        pytest.skip("PR 3: Implement after unified tokenizer")


class TestSFTTraining:
    def test_overfit_single_batch(self) -> None:
        pytest.skip("PR 3: Implement after unified tokenizer")
