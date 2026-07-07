from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "validate_data_registry.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def _make_valid_source(registry_dir: Path, name: str) -> Path:
    data = {
        "schema_version": 1,
        "source_id": name,
        "version": "1.0.0",
        "display_name": name.title(),
        "provider": "Test",
        "kind": "huggingface",
        "uri": f"https://huggingface.co/datasets/org/{name}",
        "revision": f"{name}_abc123def4567890abcdef1234567890abcdef12",
        "languages": ["en"],
        "domains": ["general"],
        "splits": ["train"],
        "purposes": ["pretraining"],
        "status": "proposed",
        "license": "mit",
        "gated": False,
        "created_at": "2025-07-07",
        "updated_at": "2025-07-07",
    }
    p = registry_dir / f"{name}.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


class TestValidateRegistryCLI:
    def test_empty_registry(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result = _run(
            "--registry-dir", str(registry_dir), "--policy", str(tmp_path / "license_policy.yaml")
        )
        assert result.returncode == 0
        assert "no sources" in result.stdout

    def test_human_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_valid_source(registry_dir, "ds1")
        _make_valid_source(registry_dir, "ds2")
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result = _run(
            "--registry-dir", str(registry_dir), "--policy", str(tmp_path / "license_policy.yaml")
        )
        assert result.returncode == 0
        assert "Total records" in result.stdout
        assert "Proposed" in result.stdout

    def test_json_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_valid_source(registry_dir, "ds1")
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result = _run(
            "--registry-dir",
            str(registry_dir),
            "--policy",
            str(tmp_path / "license_policy.yaml"),
            "--json",
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "total_records" in data
        assert "digest" in data

    def test_malformed_source_exit_code(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        p = registry_dir / "bad.yaml"
        with p.open("w") as f:
            f.write(": broken yaml\n")
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result = _run(
            "--registry-dir", str(registry_dir), "--policy", str(tmp_path / "license_policy.yaml")
        )
        assert result.returncode != 0

    def test_strict_mode_exit_code(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_valid_source(registry_dir, "ds1")
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result = _run(
            "--registry-dir",
            str(registry_dir),
            "--policy",
            str(tmp_path / "license_policy.yaml"),
            "--strict",
        )
        assert result.returncode != 0
        assert "strict mode" in result.stderr

    def test_deterministic_digest_displayed(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_valid_source(registry_dir, "ds1")
        policy_data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [],
        }
        with (tmp_path / "license_policy.yaml").open("w") as f:
            yaml.dump(policy_data, f)
        result1 = _run(
            "--registry-dir",
            str(registry_dir),
            "--policy",
            str(tmp_path / "license_policy.yaml"),
            "--json",
        )
        result2 = _run(
            "--registry-dir",
            str(registry_dir),
            "--policy",
            str(tmp_path / "license_policy.yaml"),
            "--json",
        )
        d1 = json.loads(result1.stdout)
        d2 = json.loads(result2.stdout)
        assert d1["digest"] == d2["digest"]
