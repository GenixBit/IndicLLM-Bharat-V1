from __future__ import annotations

import pytest


class TestCheckpointMetadata:
    def test_metadata_in_checkpoint(self) -> None:
        pytest.skip("PR 5: Implement after checkpoint metadata")

    def test_tokenizer_hash_stored(self) -> None:
        pytest.skip("PR 5: Implement after checkpoint metadata")

    def test_incompatible_checkpoint_rejected(self) -> None:
        pytest.skip("PR 5: Implement after checkpoint metadata")


class TestCheckpointResume:
    def test_optimizer_state_restored(self) -> None:
        pytest.skip("PR 5: Implement after checkpoint metadata")

    def test_resume_after_interruption(self) -> None:
        pytest.skip("PR 5: Implement after checkpoint metadata")
