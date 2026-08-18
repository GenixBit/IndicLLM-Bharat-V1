from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from scripts.profile_inference import (
    benchmark_single_config,
    format_markdown_table,
    main,
    parse_args,
    profile_model,
)


@pytest.fixture
def tiny_config() -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_model(tiny_config: BharatModelConfig) -> BharatForCausalLM:
    return BharatForCausalLM(tiny_config)


class TestInferenceProfiler:
    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.model_size == "350m"
        assert args.batch_sizes == "1,2"
        assert args.prompt_lengths == "64,256"
        assert args.gen_lengths == "32,64"
        assert args.device == "auto"
        assert args.dtype == "bf16"
        assert args.warmup == 2
        assert args.runs == 3
        assert args.json is False

    def test_parse_args_explicit(self):
        args = parse_args(
            [
                "--model-size",
                "tiny",
                "--batch-sizes",
                "1,4",
                "--prompt-lengths",
                "16,32",
                "--gen-lengths",
                "8",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--warmup",
                "1",
                "--runs",
                "2",
                "--json",
                "--output",
                "test_out.json",
            ]
        )
        assert args.model_size == "tiny"
        assert args.batch_sizes == "1,4"
        assert args.prompt_lengths == "16,32"
        assert args.gen_lengths == "8"
        assert args.device == "cpu"
        assert args.dtype == "fp32"
        assert args.warmup == 1
        assert args.runs == 2
        assert args.json is True
        assert args.output == "test_out.json"

    def test_benchmark_single_config(
        self, tiny_model: BharatForCausalLM, tiny_config: BharatModelConfig
    ):
        device = torch.device("cpu")
        res = benchmark_single_config(
            model=tiny_model,
            config=tiny_config,
            batch_size=2,
            prompt_len=16,
            gen_len=4,
            device=device,
            warmup=1,
            runs=1,
        )
        assert res.batch_size == 2
        assert res.prompt_length == 16
        assert res.gen_length == 4
        assert res.tokens_generated == 8
        assert res.ttft_ms > 0
        assert res.avg_itl_ms > 0
        assert res.total_time_s > 0
        assert res.gen_throughput_tok_per_s > 0
        assert res.total_throughput_tok_per_s > 0
        assert res.kv_cache_mb >= 0

    def test_profile_model(self, tiny_model: BharatForCausalLM, tiny_config: BharatModelConfig):
        report = profile_model(
            model=tiny_model,
            config=tiny_config,
            batch_sizes=[1],
            prompt_lengths=[8],
            gen_lengths=[4],
            device_name="cpu",
            dtype_name="fp32",
            warmup=1,
            runs=1,
            model_name="TestTinyModel",
        )
        assert report.model_name == "TestTinyModel"
        assert report.num_parameters > 0
        assert report.device == "cpu"
        assert len(report.results) == 1
        d = report.to_dict()
        assert "model_name" in d
        assert "results" in d
        assert len(d["results"]) == 1

    def test_format_markdown_table(
        self, tiny_model: BharatForCausalLM, tiny_config: BharatModelConfig
    ):
        report = profile_model(
            model=tiny_model,
            config=tiny_config,
            batch_sizes=[1, 2],
            prompt_lengths=[8],
            gen_lengths=[4],
            device_name="cpu",
            dtype_name="fp32",
            warmup=1,
            runs=1,
            model_name="TestTinyModel",
        )
        table = format_markdown_table(report)
        assert "Inference Profile Report: TestTinyModel" in table
        assert "TTFT (ms)" in table
        assert "Gen Throughput" in table
        assert "| 1 | 8 | 4 |" in table
        assert "| 2 | 8 | 4 |" in table

    def test_cli_tiny_json(self, capsys: pytest.CaptureFixture[str]):
        code = main(
            [
                "--model-size",
                "tiny",
                "--batch-sizes",
                "1",
                "--prompt-lengths",
                "8",
                "--gen-lengths",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--warmup",
                "1",
                "--runs",
                "1",
                "--json",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["model_name"] == "Bharat-Tiny"
        assert len(data["results"]) == 1
        assert data["results"][0]["batch_size"] == 1

    def test_cli_output_file(self, tmp_path: Path):
        out_file = tmp_path / "report.json"
        code = main(
            [
                "--model-size",
                "tiny",
                "--batch-sizes",
                "1",
                "--prompt-lengths",
                "8",
                "--gen-lengths",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "fp32",
                "--warmup",
                "1",
                "--runs",
                "1",
                "--output",
                str(out_file),
            ]
        )
        assert code == 0
        assert out_file.is_file()
        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["model_name"] == "Bharat-Tiny"
        assert len(data["results"]) == 1

    def test_cli_missing_config_returns_1(self, capsys: pytest.CaptureFixture[str]):
        code = main(["--model-config", "/nonexistent/config.yaml"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: Model config file not found" in captured.err

    def test_cli_missing_checkpoint_returns_1(self, capsys: pytest.CaptureFixture[str]):
        code = main(["--checkpoint", "/nonexistent/checkpoint.pt"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: Checkpoint file not found" in captured.err
