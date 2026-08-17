from __future__ import annotations

import json
from pathlib import Path

from bharat.data.manifest import ShardManifest, create_manifest
from scripts.plan_data_mixture import load_recipe
from scripts.plan_data_mixture import main as cli_main

ROOT = Path(__file__).resolve().parent.parent.parent
PRETRAIN_RECIPE = ROOT / "configs" / "data" / "mixture_pretrain_indic_1b.yaml"
SFT_RECIPE = ROOT / "configs" / "data" / "mixture_sft_indic.yaml"


def _create_mock_manifest(
    path: Path,
    source_id: str,
    language: str,
    domain: str,
    records: int = 5000,
) -> Path:
    manifest_file = path / f"{source_id}_{language}_{domain}" / "manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    shard = ShardManifest(
        shard_id=f"{source_id}_0000",
        index=0,
        record_start=0,
        record_end=records,
        bytes_utf8=records * 100,
        sha256="0" * 64,
        created_at="2026-01-01T00:00:00Z",
    )

    manifest = create_manifest(
        dataset_id=f"{source_id}_{language}_train",
        source_id=source_id,
        source_version="1.0.0",
        license="cc-by-4.0",
        language=language,
        split="train",
        domain=domain,
        shards=(shard,),
        records=records,
        bytes_utf8=records * 100,
        sha256="1" * 64,
        processing_config_digest="2" * 64,
        registry_digest="3" * 64,
        policy_digest="4" * 64,
        created_at="2026-01-01T00:00:00Z",
    )

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return manifest_file


class TestDataMixtureCLI:
    def test_load_official_recipes(self) -> None:
        """Verify that pretraining and SFT recipe configurations load and validate constraint sums."""
        assert PRETRAIN_RECIPE.is_file(), f"Pretrain recipe missing: {PRETRAIN_RECIPE}"
        assert SFT_RECIPE.is_file(), f"SFT recipe missing: {SFT_RECIPE}"

        pretrain_constraint = load_recipe(PRETRAIN_RECIPE)
        assert abs(sum(pretrain_constraint.language_weights.values()) - 1.0) < 1e-6
        assert abs(sum(pretrain_constraint.domain_weights.values()) - 1.0) < 1e-6

        sft_constraint = load_recipe(SFT_RECIPE)
        assert abs(sum(sft_constraint.language_weights.values()) - 1.0) < 1e-6
        assert abs(sum(sft_constraint.domain_weights.values()) - 1.0) < 1e-6

    def test_plan_data_mixture_cli_execution(self, tmp_path: Path, capsys) -> None:
        """Verify plan_data_mixture CLI runs end-to-end and outputs structured mixture plan."""
        manifests_dir = tmp_path / "manifests"
        _create_mock_manifest(manifests_dir, "indiccorp_v2", "hi", "general", records=10000)
        _create_mock_manifest(manifests_dir, "sangraha", "bn", "general", records=5000)
        _create_mock_manifest(manifests_dir, "samanantar", "ta", "general", records=4000)

        # Simple test recipe with matching languages
        test_recipe = tmp_path / "recipe.yaml"
        test_recipe.write_text(
            "recipe_name: test\n"
            "language_weights:\n"
            "  hi: 0.60\n"
            "  bn: 0.25\n"
            "  ta: 0.15\n"
            "domain_weights:\n"
            "  general: 1.0\n"
            "max_pct_per_source: 0.70\n"
            "min_record_threshold: 100\n",
            encoding="utf-8",
        )

        args = [
            "--recipe",
            str(test_recipe),
            "--manifests-dir",
            str(manifests_dir),
            "--target-records",
            "1000",
            "--json",
        ]

        ret = cli_main(args)
        assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_sources"] == 3
        assert data["total_estimated_records"] == 1000
        assert len(data["plans"]) == 3

        weights = {p["language"]: p["weight"] for p in data["plans"]}
        assert weights["hi"] == 0.60
        assert weights["bn"] == 0.25
        assert weights["ta"] == 0.15

    def test_plan_data_mixture_cli_with_target_tokens_and_output(self, tmp_path: Path) -> None:
        """Verify --target-tokens conversion and --output JSON dumping."""
        manifests_dir = tmp_path / "manifests"
        _create_mock_manifest(manifests_dir, "indiccorp_v2", "hi", "general", records=10000)
        _create_mock_manifest(manifests_dir, "sangraha", "bn", "general", records=5000)

        test_recipe = tmp_path / "recipe.yaml"
        test_recipe.write_text(
            "recipe_name: test\n"
            "language_weights:\n"
            "  hi: 0.70\n"
            "  bn: 0.30\n"
            "domain_weights:\n"
            "  general: 1.0\n"
            "max_pct_per_source: 0.80\n"
            "min_record_threshold: 100\n",
            encoding="utf-8",
        )

        output_file = tmp_path / "plan_out.json"
        args = [
            "--recipe",
            str(test_recipe),
            "--manifests-dir",
            str(manifests_dir),
            "--target-tokens",
            "512000",  # 512,000 tokens // 512 = 1,000 records
            "--output",
            str(output_file),
        ]

        ret = cli_main(args)
        assert ret == 0
        assert output_file.is_file()

        saved_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert saved_data["total_estimated_records"] == 1000
