from __future__ import annotations

from pathlib import Path

from bharat.pipeline.orchestrator import (
    PipelineConfig,
    SovereignPipelineOrchestrator,
)
from scripts.run_full_pipeline import main as pipeline_main
from scripts.run_full_pipeline import parse_args


class TestPipelineOrchestrator:
    def test_pipeline_data_and_pretrain_stages(self, tmp_path: Path):
        cfg = PipelineConfig(
            tier="tiny",
            work_dir=tmp_path / "pipe_test",
            stages=["data", "pretrain"],
            pretrain_steps=2,
            batch_size=1,
            device="cpu",
        )

        orchestrator = SovereignPipelineOrchestrator(cfg)
        manifest = orchestrator.run_pipeline()

        assert manifest.tier == "tiny"
        assert len(manifest.stages) == 2
        assert manifest.stages[0]["status"] == "SUCCESS"
        assert manifest.stages[1]["status"] == "SUCCESS"
        assert (tmp_path / "pipe_test" / "pipeline_manifest.json").is_file()

    def test_pipeline_export_and_eval_stages(self, tmp_path: Path):
        cfg = PipelineConfig(
            tier="tiny",
            work_dir=tmp_path / "pipe_eval_test",
            stages=["export", "eval"],
            device="cpu",
        )

        orchestrator = SovereignPipelineOrchestrator(cfg)
        manifest = orchestrator.run_pipeline()

        assert len(manifest.stages) == 2
        assert (tmp_path / "pipe_eval_test" / "bharat_edge_q8_0.gguf").is_file()
        assert (tmp_path / "pipe_eval_test" / "eval_report.json").is_file()

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "1b", "--stages", "data,pretrain", "--pretrain-steps", "10"])
        assert args.tier == "1b"
        assert args.stages == "data,pretrain"
        assert args.pretrain_steps == 10

    def test_cli_main(self, tmp_path: Path):
        code = pipeline_main(
            [
                "--tier",
                "tiny",
                "--stages",
                "data",
                "--work-dir",
                str(tmp_path / "cli_pipe"),
                "--device",
                "cpu",
            ]
        )
        assert code == 0
        assert (tmp_path / "cli_pipe" / "pipeline_manifest.json").is_file()
