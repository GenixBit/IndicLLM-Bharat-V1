from __future__ import annotations

import json
from pathlib import Path

from bharat.eval.leaderboard import (
    build_default_sovereign_leaderboard,
    export_leaderboard_files,
    format_markdown_leaderboard,
)
from scripts.generate_leaderboard import main as leaderboard_main
from scripts.generate_leaderboard import parse_args


class TestLeaderboard:
    def test_build_default_leaderboard(self):
        report = build_default_sovereign_leaderboard()
        assert len(report.entries) >= 6
        assert any(e.tier == "10B" for e in report.entries)
        assert any(e.stage == "GGUF Q8_0" for e in report.entries)

    def test_format_markdown_leaderboard(self):
        report = build_default_sovereign_leaderboard()
        md = format_markdown_leaderboard(report)
        assert "# 🏆 IndicLLM-Bharat Cross-Tier Sovereign Leaderboard" in md
        assert "| Rank | Model Name |" in md
        assert "Bharat-10B-DPO" in md

    def test_export_leaderboard_files(self, tmp_path: Path):
        report = build_default_sovereign_leaderboard()
        exported = export_leaderboard_files(report, tmp_path)

        assert "markdown" in exported and exported["markdown"].is_file()
        assert "json" in exported and exported["json"].is_file()

        data = json.loads(exported["json"].read_text(encoding="utf-8"))
        assert "entries" in data
        assert len(data["entries"]) == len(report.entries)

    def test_cli_parse_args(self):
        args = parse_args(["--output-dir", "custom_leaderboard"])
        assert args.output_dir == "custom_leaderboard"

    def test_cli_main(self, tmp_path: Path):
        code = leaderboard_main(["--output-dir", str(tmp_path)])
        assert code == 0
        assert (tmp_path / "LEADERBOARD.md").is_file()
        assert (tmp_path / "leaderboard.json").is_file()
