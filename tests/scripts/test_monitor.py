from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.monitor import (
    check_checkpoints,
    check_data_pipeline,
    check_eval_benchmarks,
    get_hardware_telemetry,
)
from scripts.monitor import main as monitor_cli_main


class TestMonitor:
    def test_get_hardware_telemetry(self) -> None:
        telem = get_hardware_telemetry()
        assert "platform" in telem
        assert "cpu_count" in telem
        assert telem["cpu_count"] >= 1
        assert "device" in telem
        assert telem["device"] in ("cpu", "cuda", "mps")

    def test_check_checkpoints(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        f1 = ckpt_dir / "model_step_10.pt"
        torch.save({"step": 10}, f1)

        ckpts = check_checkpoints(ckpt_dir)
        assert len(ckpts) == 1
        assert ckpts[0]["file"] == "model_step_10.pt"
        assert ckpts[0]["size_mb"] >= 0

    def test_check_data_pipeline(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        shard = data_dir / "train_0000.bin"
        shard.write_bytes(b"\x00" * 1024)

        manifest = data_dir / "manifest.json"
        manifest.write_text('{"count": 10}', encoding="utf-8")

        stats = check_data_pipeline(data_dir)
        assert stats["exists"] is True
        assert stats["total_shards"] == 1
        assert stats["manifest_count"] == 1

    def test_check_eval_benchmarks(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval_out"
        eval_dir.mkdir()
        rep = eval_dir / "report_bharat.json"
        rep.write_text(
            json.dumps(
                {
                    "model_name": "Bharat-350M",
                    "aggregate_score": 0.85,
                    "total_examples": 100,
                }
            ),
            encoding="utf-8",
        )

        ev = check_eval_benchmarks(eval_dir)
        assert ev["exists"] is True
        assert len(ev["runs"]) == 1
        assert ev["runs"][0]["model_name"] == "Bharat-350M"
        assert ev["runs"][0]["aggregate_score"] == 0.85

    def test_monitor_cli_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ret = monitor_cli_main(
            [
                "--checkpoints-dir",
                str(tmp_path),
                "--data-dir",
                str(tmp_path),
                "--eval-dir",
                str(tmp_path),
                "--json",
            ]
        )
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "telemetry" in data
        assert "checkpoints" in data
