from __future__ import annotations

import pytest


class TestDPODataset:
    def test_dataset_initialization(self) -> None:
        pytest.skip("Needs tokenizer fixture")

    def test_padding(self) -> None:
        pytest.skip("Needs tokenizer fixture")


class TestDPOLoss:
    def test_dpo_loss_initialization(self) -> None:
        pytest.skip("PR 4: Implement after unified tokenizer")

    def test_per_sample_masking(self) -> None:
        pytest.skip("PR 4: Implement after unified tokenizer")

    def test_variable_prompt_lengths(self) -> None:
        pytest.skip("PR 4: Implement after unified tokenizer")


class TestDPOTraining:
    def test_overfit_single_batch(self) -> None:
        pytest.skip("PR 4: Implement after unified tokenizer")
