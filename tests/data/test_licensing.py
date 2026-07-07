from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.licensing import (
    LicenseDecision,
    load_license_policy,
)


def _policy_path(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "policy.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


class TestLicensing:
    def test_default_deny(self, tmp_path):
        policy = load_license_policy(
            _policy_path(
                tmp_path,
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [],
                },
            )
        )
        assert policy.default_decision == LicenseDecision.DENY

    def test_missing_license_rejected(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("unknown_license") == LicenseDecision.DENY

    def test_unknown_identifier_rejected_for_approval(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("nonexistent") == LicenseDecision.DENY

    def test_allow_requires_evidence_and_verification(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [
                    {
                        "identifier": "mit",
                        "name": "MIT License",
                        "decision": "allow",
                        "evidence_url": "https://example.com/mit",
                        "verified_at": "2025-07-01",
                        "verified_by": "project_team",
                        "commercial_use_allowed": True,
                        "model_training_allowed": True,
                        "redistribution_allowed": True,
                    }
                ],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("mit") == LicenseDecision.ALLOW
        lic = policy.resolve("mit")
        assert lic is not None
        assert lic.evidence_url == "https://example.com/mit"
        assert lic.verified_at == "2025-07-01"
        assert lic.verified_by == "project_team"

    def test_non_commercial_restriction_not_silently_allowed(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [
                    {
                        "identifier": "cc_by_nc",
                        "name": "CC BY-NC 4.0",
                        "decision": "review",
                        "commercial_use_allowed": False,
                        "model_training_allowed": False,
                    }
                ],
            },
        )
        policy = load_license_policy(p)
        lic = policy.resolve("cc_by_nc")
        assert lic is not None
        assert lic.commercial_use_allowed is False

    def test_custom_terms_require_review(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "review",
                "licenses": [
                    {
                        "identifier": "custom_v1",
                        "name": "Custom Terms v1",
                        "decision": "review",
                        "notes": "Requires project review",
                    }
                ],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("custom_v1") == LicenseDecision.REVIEW

    def test_deny_cannot_become_approved(self, tmp_path):
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [
                    {
                        "identifier": "restrictive",
                        "name": "Restrictive License",
                        "decision": "deny",
                    }
                ],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("restrictive") == LicenseDecision.DENY

    def test_policy_schema_validation_missing_version(self, tmp_path):
        p = tmp_path / "policy.yaml"
        with p.open("w") as f:
            yaml.dump({"default_decision": "deny"}, f)
        with pytest.raises((ValueError, TypeError)):
            load_license_policy(p)

    def test_policy_schema_validation_unknown_root_key(self, tmp_path):
        p = tmp_path / "policy.yaml"
        with p.open("w") as f:
            yaml.dump(
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [],
                    "extra_key": "value",
                },
                f,
            )
        with pytest.raises(ValueError, match="unknown root key"):
            load_license_policy(p)

    def test_duplicate_license_identifier_rejected(self, tmp_path):
        p = tmp_path / "policy.yaml"
        with p.open("w") as f:
            yaml.dump(
                {
                    "schema_version": 1,
                    "default_decision": "deny",
                    "licenses": [
                        {"identifier": "mit", "name": "MIT", "decision": "allow"},
                        {"identifier": "mit", "name": "MIT Duplicate", "decision": "allow"},
                    ],
                },
                f,
            )
        with pytest.raises(ValueError, match="duplicate license identifier"):
            load_license_policy(p)

    def test_review_cannot_become_approved(self, tmp_path):
        """REVIEW licences must not be treated as approved."""
        p = _policy_path(
            tmp_path,
            {
                "schema_version": 1,
                "default_decision": "deny",
                "licenses": [
                    {
                        "identifier": "unreviewed",
                        "name": "Unreviewed",
                        "decision": "review",
                    }
                ],
            },
        )
        policy = load_license_policy(p)
        assert policy.decision_for("unreviewed") == LicenseDecision.REVIEW
