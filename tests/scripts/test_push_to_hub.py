from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.build_release_bundle import build_release_bundle
from scripts.push_to_hub import main as push_cli_main
from scripts.push_to_hub import push_to_hub


@pytest.fixture
def dummy_bundle(tmp_path: Path) -> Path:
    cp_file = tmp_path / "model.pt"
    t1 = torch.randn(32, 32, dtype=torch.float32)
    state = {"model": {"weight": t1}}
    torch.save(state, cp_file)

    bundle_dir = tmp_path / "dist_bundle"
    build_release_bundle(
        checkpoint_path=cp_file,
        output_dir=bundle_dir,
        model_name="Bharat-Test",
        version="1.0.0",
    )
    return bundle_dir


@pytest.fixture
def dummy_gpt2_checkpoint(tmp_path: Path) -> Path:
    cp_file = tmp_path / "gpt2.pt"
    state = {
        "model": {
            "wte.weight": torch.randn(50257, 64),
            "wpe.weight": torch.randn(128, 64),
            "h.0.ln_1.weight": torch.randn(64),
            "h.0.ln_1.bias": torch.randn(64),
            "h.0.attn.c_attn.weight": torch.randn(192, 64),
            "h.0.attn.c_attn.bias": torch.randn(192),
            "h.0.attn.c_proj.weight": torch.randn(64, 64),
            "h.0.attn.c_proj.bias": torch.randn(64),
            "h.0.ln_2.weight": torch.randn(64),
            "h.0.ln_2.bias": torch.randn(64),
            "h.0.mlp.c_fc.weight": torch.randn(256, 64),
            "h.0.mlp.c_fc.bias": torch.randn(256),
            "h.0.mlp.c_proj.weight": torch.randn(64, 256),
            "h.0.mlp.c_proj.bias": torch.randn(64),
            "ln_f.weight": torch.randn(64),
            "ln_f.bias": torch.randn(64),
        },
        "config": {
            "vocab_size": 50257,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 128,
        },
    }
    torch.save(state, cp_file)
    return cp_file


class TestPushToHub:
    def test_dry_run_with_bundle_dir(self, dummy_bundle: Path) -> None:
        result = push_to_hub(
            repo_id="GenixBit/IndicLLM-Test",
            bundle_dir=dummy_bundle,
            model_name="Bharat-Test",
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["repo_id"] == "GenixBit/IndicLLM-Test"
        assert result["files_count"] >= 4
        assert "model.safetensors" in result["files"]
        assert "README.md" in result["files"]
        assert "config.json" in result["files"]

    def test_dry_run_cli(self, dummy_bundle: Path) -> None:
        ret = push_cli_main(
            [
                "--bundle-dir",
                str(dummy_bundle),
                "--repo",
                "GenixBit/IndicLLM-CLI-Test",
                "--dry-run",
            ]
        )
        assert ret == 0

    @patch("huggingface_hub.HfApi")
    def test_mocked_upload_from_bundle(
        self, mock_hf_api_class: MagicMock, dummy_bundle: Path
    ) -> None:
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        result = push_to_hub(
            repo_id="GenixBit/IndicLLM-Mock-Test",
            bundle_dir=dummy_bundle,
            private=True,
            commit_msg="Test release commit",
            dry_run=False,
        )

        assert result["dry_run"] is False
        mock_api_instance.create_repo.assert_called_once_with(
            repo_id="GenixBit/IndicLLM-Mock-Test",
            private=True,
            exist_ok=True,
        )
        mock_api_instance.upload_folder.assert_called_once()
        call_kwargs = mock_api_instance.upload_folder.call_args.kwargs
        assert call_kwargs["repo_id"] == "GenixBit/IndicLLM-Mock-Test"
        assert call_kwargs["commit_message"] == "Test release commit"

    def test_dry_run_gpt2_checkpoint_conversion(self, dummy_gpt2_checkpoint: Path) -> None:
        result = push_to_hub(
            repo_id="GenixBit/IndicLLM-GPT2-Test",
            checkpoint_path=dummy_gpt2_checkpoint,
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert "config.json" in result["files"]
        assert "README.md" in result["files"]
