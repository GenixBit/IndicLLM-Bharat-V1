from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from inference.export_ollama import export_ollama, generate_modelfile
from inference.export_ollama import main as export_cli_main


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    cp_file = tmp_path / "model.pt"
    # Ensure dimensions are multiples of 32 for GGUF Q8_0
    t1 = torch.randn(32, 32, dtype=torch.float32)
    t2 = torch.randn(32, 64, dtype=torch.float32)
    state = {
        "model": {
            "model.embed_tokens.weight": t1,
            "lm_head.weight": t2,
        },
        "step": 100,
    }
    torch.save(state, cp_file)
    return cp_file


class TestExportOllama:
    def test_generate_modelfile(self, tmp_path: Path) -> None:
        mf = generate_modelfile(
            gguf_filename="model-q8_0.gguf",
            output_dir=tmp_path,
            context_length=2048,
        )
        assert mf.is_file()
        content = mf.read_text(encoding="utf-8")
        assert "FROM ./model-q8_0.gguf" in content
        assert "PARAMETER num_ctx 2048" in content
        assert 'stop "<|endoftext|>"' in content

    def test_export_q8_0_and_modelfile(self, tmp_path: Path, dummy_checkpoint: Path) -> None:
        out_dir = tmp_path / "ollama_export"
        res = export_ollama(
            checkpoint_path=dummy_checkpoint,
            output_dir=out_dir,
            name="bharat-test",
            quant="q8_0",
            register=False,
        )

        assert Path(res["gguf_path"]).is_file()
        assert Path(res["modelfile_path"]).is_file()
        assert res["name"] == "bharat-test"
        assert res["quant"] == "q8_0"

    def test_export_f32_and_modelfile(self, tmp_path: Path, dummy_checkpoint: Path) -> None:
        out_dir = tmp_path / "ollama_f32"
        res = export_ollama(
            checkpoint_path=dummy_checkpoint,
            output_dir=out_dir,
            name="bharat-f32",
            quant="f32",
            register=False,
        )

        assert Path(res["gguf_path"]).is_file()
        assert Path(res["modelfile_path"]).is_file()
        assert res["quant"] == "f32"

    def test_export_cli(self, tmp_path: Path, dummy_checkpoint: Path) -> None:
        out_dir = tmp_path / "cli_out"
        ret = export_cli_main(
            [
                "--checkpoint",
                str(dummy_checkpoint),
                "--output-dir",
                str(out_dir),
                "--name",
                "bharat-cli",
                "--quant",
                "q8_0",
            ]
        )
        assert ret == 0
        assert (out_dir / "bharat-cli-q8_0.gguf").is_file()
        assert (out_dir / "Modelfile").is_file()

    @patch("shutil.which", return_value="/usr/local/bin/ollama")
    @patch("subprocess.run")
    def test_export_with_mocked_ollama_registration(
        self, mock_run, mock_which, tmp_path: Path, dummy_checkpoint: Path
    ) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""

        out_dir = tmp_path / "mock_reg"
        res = export_ollama(
            checkpoint_path=dummy_checkpoint,
            output_dir=out_dir,
            name="bharat-reg",
            register=True,
        )

        assert res["registered"] is True
        mock_run.assert_called_once()
