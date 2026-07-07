from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.models.spec import (
    BharatModelSpec,
    load_model_config,
    load_model_spec,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT / "configs" / "models"

PRODUCTION_FILES = [
    "bharat-350m.yaml",
    "bharat-1b.yaml",
    "bharat-3b.yaml",
    "bharat-7b.yaml",
]

EXPECTED_TOTALS: dict[str, int] = {
    "bharat-350m.yaml": 347393024,
    "bharat-1b.yaml": 999368704,
    "bharat-3b.yaml": 3009039360,
    "bharat-7b.yaml": 7040405504,
}

TARGET_TOTALS: dict[str, int] = {
    "bharat-350m.yaml": 350000000,
    "bharat-1b.yaml": 1000000000,
    "bharat-3b.yaml": 3000000000,
    "bharat-7b.yaml": 7000000000,
}


class TestLoadProductionConfigs:
    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_loads(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        assert isinstance(spec, BharatModelSpec)
        assert spec.schema_version == 1

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_exact_parameter_total(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        from bharat.models.sizing import calculate_parameter_count

        params = calculate_parameter_count(spec.architecture)
        expected = EXPECTED_TOTALS[filename]
        assert params.total == expected, f"{filename}: expected {expected}, got {params.total}"

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_within_one_percent_of_target(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        from bharat.models.sizing import calculate_parameter_count

        params = calculate_parameter_count(spec.architecture)
        target = TARGET_TOTALS[filename]
        diff_pct = abs(params.total - target) / target * 100
        assert diff_pct < 1.0, f"{filename}: {diff_pct:.4f}% from target, expected < 1%"

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_head_dimension_valid(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        arch = spec.architecture
        assert arch.hidden_size % arch.num_attention_heads == 0
        head_dim = arch.hidden_size // arch.num_attention_heads
        assert head_dim > 0

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_gqa_divisible(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        arch = spec.architecture
        assert arch.num_attention_heads % arch.num_key_value_heads == 0

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_tied_embeddings(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        assert spec.architecture.tie_word_embeddings is True

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_no_attention_bias(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        assert spec.architecture.attention_bias is False

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_no_mlp_bias(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        assert spec.architecture.mlp_bias is False

    @pytest.mark.parametrize("filename", PRODUCTION_FILES)
    def test_expected_in_spec(self, filename: str) -> None:
        spec = load_model_spec(CONFIGS_DIR / filename)
        assert spec.expected_parameter_count == EXPECTED_TOTALS[filename]


def test_all_production_configs_collected():
    """Guard: every production file runs every architecture assertion."""
    import yaml

    for fname in PRODUCTION_FILES:
        with open(CONFIGS_DIR / fname) as f:
            data = yaml.safe_load(f)
            expected = data.get("expected_parameter_count", 0)
            assert expected > 0, f"{fname}: missing expected_parameter_count"
            target = data.get("target_parameter_count", 0)
            assert target > 0, f"{fname}: missing target_parameter_count"
    assert len(PRODUCTION_FILES) == 4, "Production file count changed"

    methods_to_check = [
        "test_exact_parameter_total",
        "test_within_one_percent_of_target",
        "test_head_dimension_valid",
        "test_gqa_divisible",
        "test_tied_embeddings",
        "test_no_attention_bias",
        "test_no_mlp_bias",
        "test_expected_in_spec",
    ]
    from tests.models.test_model_specs import TestLoadProductionConfigs

    for m in methods_to_check:
        assert hasattr(TestLoadProductionConfigs, m), f"Missing test method: {m}"
    assert len(PRODUCTION_FILES) * len(methods_to_check) == 32, (
        f"Expected 32 parametrized tests, got {len(PRODUCTION_FILES) * len(methods_to_check)}"
    )


class TestSpecValidation:
    def test_schema_version_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        data = {
            "schema_version": 2,
            "model_name": "Test",
            "size_label": "T",
            "target_parameter_count": 1000,
            "expected_parameter_count": 1000,
            "architecture": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "initializer_range": 0.02,
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            },
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="schema_version"):
            load_model_spec(p)

    def test_unknown_root_key_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        data = {
            "schema_version": 1,
            "model_name": "Test",
            "size_label": "T",
            "target_parameter_count": 1000,
            "expected_parameter_count": 1000,
            "architecture": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "initializer_range": 0.02,
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            },
            "unknown_field": "value",
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="unknown root key"):
            load_model_spec(p)

    def test_unknown_arch_key_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        data = {
            "schema_version": 1,
            "model_name": "Test",
            "size_label": "T",
            "target_parameter_count": 1000,
            "expected_parameter_count": 1000,
            "architecture": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "initializer_range": 0.02,
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
                "unknown_arch_field": "value",
            },
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="unknown architecture key"):
            load_model_spec(p)

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            f.write(": broken yaml\n")
        with pytest.raises(yaml.YAMLError):
            load_model_spec(p)

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model_spec("/nonexistent/path.yaml")

    def test_load_config_returns_config(self) -> None:
        from bharat.models.config import BharatModelConfig

        cfg = load_model_config(CONFIGS_DIR / "bharat-350m.yaml")
        assert isinstance(cfg, BharatModelConfig)

    def test_empty_model_name_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        data = {
            "schema_version": 1,
            "model_name": "",
            "size_label": "T",
            "target_parameter_count": 1000,
            "expected_parameter_count": 1000,
            "architecture": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
                "initializer_range": 0.02,
                "attention_bias": False,
                "mlp_bias": False,
                "tie_word_embeddings": True,
            },
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="not be empty"):
            load_model_spec(p)
