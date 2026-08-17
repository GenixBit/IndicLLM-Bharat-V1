from __future__ import annotations

import json
from pathlib import Path

from bharat.data.licensing import load_license_policy
from bharat.data.registry import DataRegistry
from bharat.data.sources import load_source_spec
from scripts.prepare_indic_data import main as cli_main

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_DIR = ROOT / "data_registry" / "sources"
POLICY_PATH = ROOT / "data_registry" / "license_policy.yaml"


class TestIndicPipelineUnification:
    def test_indic_registry_sources_exist_and_validate(self) -> None:
        """Verify that all standardized Indic source specifications exist and validate cleanly."""
        assert SOURCES_DIR.is_dir(), f"Sources directory missing: {SOURCES_DIR}"
        assert POLICY_PATH.is_file(), f"License policy file missing: {POLICY_PATH}"

        policy = load_license_policy(POLICY_PATH)
        assert policy.schema_version == 1

        registry = DataRegistry.load(
            str(SOURCES_DIR),
            policy_path=str(POLICY_PATH),
        )

        expected_sources = {"indiccorp_v2", "sangraha", "samanantar", "wikipedia_indic"}
        loaded_sources = {spec.source_id for spec in registry.list_all()}

        assert expected_sources.issubset(
            loaded_sources
        ), f"Missing expected Indic sources: {expected_sources - loaded_sources}"

        for source_id in expected_sources:
            spec_file = SOURCES_DIR / f"{source_id}.yaml"
            spec = load_source_spec(spec_file)
            assert spec.source_id == source_id
            assert len(spec.languages) >= 1
            record = policy.resolve(spec.license)
            assert record is not None
            assert record.decision.value == "allow"

    def test_prepare_indic_data_cli_execution(self, tmp_path: Path) -> None:
        """Verify unified Indic data preparation CLI creates valid shards and manifest."""
        input_file = tmp_path / "raw_indic.txt"
        lines = [
            "भारत एक विशाल और सुंदर देश है जिसमें कई भाषाएँ बोली जाती हैं।",
            "বাংলা भाषा भारतीय उपमहाद्वीप की प्रमुख भाषा है।",
            "தமிழ் மொழி மிக பழமையான மற்றும் சிறப்பான மொழிகளில் ஒன்றாகும்.",
            "Artificial intelligence and large language models are transforming computing across India.",
            "छोटा",  # short doc (< 10 chars) to test quality filter rejection
        ]
        input_file.write_text("\n".join(lines), encoding="utf-8")

        output_dir = tmp_path / "output_shards"
        args = [
            "--input",
            str(input_file),
            "--source-id",
            "indiccorp_v2",
            "--language",
            "hi",
            "--sources-dir",
            str(SOURCES_DIR),
            "--output-dir",
            str(output_dir),
            "--max-records-per-shard",
            "2",
            "--json",
        ]

        ret = cli_main(args)
        assert ret == 0

        manifest_file = output_dir / "manifest.json"
        assert manifest_file.is_file()

        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest_data["source_id"] == "indiccorp_v2"
        assert manifest_data["language"] == "hi"
        assert manifest_data["records"] > 0
        assert len(manifest_data["shards"]) > 0

    def test_prepare_indic_data_dry_run(self, tmp_path: Path, capsys) -> None:
        """Verify dry-run mode computes statistics without writing shards."""
        input_file = tmp_path / "sample.txt"
        input_file.write_text(
            "नमस्ते भारत यह एक डेटा प्रोसेसिंग पाइपलाइन परीक्षण है जिसमें कई शब्द हैं।\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "dry_run_out"
        args = [
            "--input",
            str(input_file),
            "--source-id",
            "indiccorp_v2",
            "--language",
            "hi",
            "--sources-dir",
            str(SOURCES_DIR),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--json",
        ]

        ret = cli_main(args)
        assert ret == 0
        assert not output_dir.exists()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_records"] == 1
        assert data["accepted_records"] == 1
        assert data["shard_count"] == 0
