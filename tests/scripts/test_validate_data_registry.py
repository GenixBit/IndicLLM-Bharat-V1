from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.validate_data_registry", *args],
        capture_output=True,
        text=True,
    )


def _make_source(dir_path: Path, name: str, overrides: dict | None = None) -> Path:
    data = {
        "schema_version": 1,
        "source_id": name,
        "version": "1.0.0",
        "display_name": name,
        "provider": "test",
        "kind": "http",
        "uri": f"https://example.com/{name}",
        "revision": "abc123def456abc123def456abc123def456abc1",
        "languages": ["en"],
        "domains": ["general"],
        "splits": ["train"],
        "purposes": ["pretraining"],
        "status": "proposed",
        "license": "cc-by-4.0",
        "created_at": "2025-01-01",
        "updated_at": "2025-06-01",
    }
    if overrides:
        data.update(overrides)
    p = dir_path / f"{name}.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


def _policy_path(policy_dir: Path) -> Path:
    data = {
        "schema_version": 1,
        "default_decision": "deny",
        "licenses": [
            {
                "identifier": "cc-by-4.0",
                "name": "CC BY 4.0",
                "decision": "allow",
                "evidence_url": "https://example.com/cc-by-4.0",
                "verified_at": "2025-06-01",
                "verified_by": "project_team",
                "commercial_use_allowed": True,
                "model_training_allowed": True,
                "redistribution_allowed": True,
                "attribution_required": True,
                "share_alike": False,
            }
        ],
    }
    p = policy_dir / "license_policy.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


class TestValidateRegistryCLI:
    def test_empty_registry(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy)])
        assert result.returncode == 0
        assert "no sources" in result.stdout
        assert "Registry digest" in result.stdout
        assert "Policy digest" in result.stdout

    def test_json_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1")
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "valid"
        assert data["total_records"] == 1
        assert data["proposed_count"] == 1
        assert "registry_digest" in data
        assert "policy_digest" in data
        assert data["errors"] == []
        assert data["warnings"] == []

    def test_strict_mode_proposed_fails(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1")
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        ok = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy)])
        assert ok.returncode == 0
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--strict"])
        assert result.returncode != 0
        assert "strict mode" in result.stdout

    def test_strict_mode_review_licence_fails(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1", overrides={"license": "review-license"})
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        p = policy_dir / "license_policy.yaml"
        data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [
                {
                    "identifier": "review-license",
                    "name": "Review License",
                    "decision": "review",
                }
            ],
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(p), "--strict"])
        assert result.returncode != 0
        assert "strict mode" in result.stdout

    def test_strict_mode_denied_licence_fails(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1", overrides={"license": "restricted-license"})
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        p = policy_dir / "license_policy.yaml"
        data = {
            "schema_version": 1,
            "default_decision": "deny",
            "licenses": [
                {
                    "identifier": "restricted-license",
                    "name": "Restricted License",
                    "decision": "deny",
                }
            ],
        }
        with p.open("w") as f:
            yaml.dump(data, f)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(p), "--strict"])
        assert result.returncode != 0
        assert "strict mode" in result.stdout

    def test_json_failure_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(
            registry_dir,
            "ds1",
            overrides={
                "status": "approved",
                "license": "unknown-license",
                "integrity": {
                    "revision": "abc123def456abc123def456abc123def456abc1",
                    "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",
                },
            },
        )
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--json"])
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert data["status"] == "invalid"
        assert len(data["errors"]) > 0

    def test_policy_digest_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1")
        _make_source(registry_dir, "ds2")
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        result = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["registry_digest"]
        assert data["policy_digest"]
        assert data["registry_digest"] != data["policy_digest"]

    def test_deterministic_output(self, tmp_path):
        registry_dir = tmp_path / "sources"
        registry_dir.mkdir()
        _make_source(registry_dir, "ds1")
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        policy = _policy_path(policy_dir)
        r1 = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--json"])
        r2 = run_cli(["--registry-dir", str(registry_dir), "--policy", str(policy), "--json"])
        d1 = json.loads(r1.stdout)
        d2 = json.loads(r2.stdout)
        assert d1["registry_digest"] == d2["registry_digest"]
        assert d1["policy_digest"] == d2["policy_digest"]
