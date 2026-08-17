from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.build_release_bundle import build_release_bundle, compute_sha256
from scripts.build_release_bundle import main as bundle_cli_main
from scripts.sanity_check import main as sanity_cli_main


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    cp_file = tmp_path / "model.pt"
    # Create valid 2D tensor state
    t1 = torch.randn(32, 32, dtype=torch.float32)
    t2 = torch.randn(32, 64, dtype=torch.float32)
    state = {
        "model": {
            "weight_1": t1,
            "weight_2": t2,
        },
        "step": 100,
    }
    torch.save(state, cp_file)
    return cp_file


class TestReleaseBundleBuilder:
    def test_build_release_bundle_safetensors_and_manifest(
        self, tmp_path: Path, dummy_checkpoint: Path
    ) -> None:
        out_dir = tmp_path / "release_v1"
        model_card = tmp_path / "MODEL_CARD.md"
        model_card.write_text("# Test Model Card", encoding="utf-8")

        manifest = build_release_bundle(
            checkpoint_path=dummy_checkpoint,
            output_dir=out_dir,
            model_name="Bharat-Test-350M",
            version="1.0.0",
            include_gguf=False,
            model_card_path=model_card,
        )

        assert manifest["model_name"] == "Bharat-Test-350M"
        assert manifest["version"] == "1.0.0"
        assert manifest["total_files"] >= 3

        # Verify files exist on disk
        assert (out_dir / "model.safetensors").is_file()
        assert (out_dir / "config.json").is_file()
        assert (out_dir / "MODEL_CARD.md").is_file()
        assert (out_dir / "release_manifest.json").is_file()

        # Verify cryptographic integrity
        for file_info in manifest["files"]:
            target_path = out_dir / file_info["filename"]
            assert target_path.is_file()
            assert target_path.stat().st_size == file_info["size_bytes"]
            assert compute_sha256(target_path) == file_info["sha256"]

    def test_build_release_bundle_with_gguf(self, tmp_path: Path, dummy_checkpoint: Path) -> None:
        out_dir = tmp_path / "release_gguf"
        manifest = build_release_bundle(
            checkpoint_path=dummy_checkpoint,
            output_dir=out_dir,
            model_name="Bharat-GGUF",
            version="1.1.0",
            include_gguf=True,
            gguf_type="Q8_0",
        )

        assert (out_dir / "model-q8_0.gguf").is_file()
        filenames = [f["filename"] for f in manifest["files"]]
        assert "model-q8_0.gguf" in filenames
        assert "model.safetensors" in filenames

    def test_bundle_cli_execution(self, tmp_path: Path, dummy_checkpoint: Path, capsys) -> None:
        out_dir = tmp_path / "cli_bundle"
        args = [
            "--checkpoint",
            str(dummy_checkpoint),
            "--output-dir",
            str(out_dir),
            "--model-name",
            "Bharat-CLI",
            "--version",
            "2.0.0",
            "--json",
        ]
        ret = bundle_cli_main(args)
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["model_name"] == "Bharat-CLI"
        assert data["version"] == "2.0.0"

    def test_sanity_check_cli_bharat(self) -> None:
        ret = sanity_cli_main(["--model", "bharat", "--max-iters", "2", "--device", "cpu"])
        assert ret == 0
