from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bharat.data.licensing import (
    LicenseDecision,
    LicensePolicy,
    load_license_policy,
)


def _policy_path(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "policy.yaml"
    with p.open("w") as f:
        yaml.dump(data, f)
    return p


def _make_allow_dict(overrides: dict | None = None) -> dict:
    base = {
        "identifier": "mit",
        "name": "MIT License",
        "decision": "allow",
        "evidence_url": "https://example.com/mit",
        "verified_at": "2025-07-01",
        "verified_by": "project_team",
        "commercial_use_allowed": True,
        "model_training_allowed": True,
        "redistribution_allowed": True,
        "attribution_required": True,
        "share_alike": False,
    }
    if overrides:
        base.update(overrides)
    return base


def _policy_with_licenses(licenses: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "default_decision": "deny",
        "licenses": licenses,
    }


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

    def test_default_allow_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="default_decision must be 'deny', got 'allow'"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    {
                        "schema_version": 1,
                        "default_decision": "allow",
                        "licenses": [],
                    },
                )
            )

    def test_default_review_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="default_decision must be 'deny', got 'review'"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    {
                        "schema_version": 1,
                        "default_decision": "review",
                        "licenses": [],
                    },
                )
            )

    def test_missing_identifier_resolves_to_deny(self, tmp_path):
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
        assert policy.decision_for("unknown_license") == LicenseDecision.DENY

    def test_unknown_identifier_resolves_to_deny(self, tmp_path):
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
        assert policy.decision_for("nonexistent") == LicenseDecision.DENY

    def test_allow_without_evidence_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="evidence_url is required"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"evidence_url": None})]),
                )
            )

    def test_allow_without_verified_by_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="verified_by is required"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"verified_by": None})]),
                )
            )

    def test_allow_without_verified_at_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="verified_at is required"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"verified_at": None})]),
                )
            )

    def test_allow_with_commercial_use_false_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="commercial_use_allowed must be true"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"commercial_use_allowed": False})]),
                )
            )

    def test_allow_with_model_training_false_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="model_training_allowed must be true"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"model_training_allowed": False})]),
                )
            )

    def test_allow_missing_attribution_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="attribution_required is required"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"attribution_required": None})]),
                )
            )

    def test_allow_missing_share_alike_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="share_alike is required"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"share_alike": None})]),
                )
            )

    def test_allow_with_https_evidence_url(self, tmp_path):
        policy = load_license_policy(
            _policy_path(
                tmp_path,
                _policy_with_licenses([_make_allow_dict()]),
            )
        )
        assert policy.decision_for("mit") == LicenseDecision.ALLOW
        lic = policy.resolve("mit")
        assert lic is not None
        assert lic.evidence_url == "https://example.com/mit"
        assert lic.verified_at == "2025-07-01"
        assert lic.verified_by == "project_team"
        assert lic.attribution_required is True
        assert lic.share_alike is False

    def test_malformed_evidence_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="evidence_url must be a valid https:// URL"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses(
                        [_make_allow_dict({"evidence_url": "http://example.com"})]
                    ),
                )
            )

    def test_invalid_verification_date_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="verified_at must be a valid ISO-8601 date"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict({"verified_at": "not-a-date"})]),
                )
            )

    def test_unknown_id_under_policy_with_default_allow(self, tmp_path):
        policy = LicensePolicy(
            schema_version=1,
            default_decision=LicenseDecision.ALLOW,
            licenses=(),
        )
        assert policy.decision_for("nonexistent") == LicenseDecision.DENY

    def test_review_cannot_become_approved(self, tmp_path):
        policy = load_license_policy(
            _policy_path(
                tmp_path,
                _policy_with_licenses(
                    [
                        {
                            "identifier": "unreviewed",
                            "name": "Unreviewed",
                            "decision": "review",
                        }
                    ]
                ),
            )
        )
        assert policy.decision_for("unreviewed") == LicenseDecision.REVIEW

    def test_deny_cannot_become_approved(self, tmp_path):
        policy = load_license_policy(
            _policy_path(
                tmp_path,
                _policy_with_licenses(
                    [
                        {
                            "identifier": "restrictive",
                            "name": "Restrictive License",
                            "decision": "deny",
                        }
                    ]
                ),
            )
        )
        assert policy.decision_for("restrictive") == LicenseDecision.DENY

    def test_policy_schema_validation_missing_version(self, tmp_path):
        with pytest.raises(TypeError, match="schema_version must be an integer"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    {
                        "default_decision": "deny",
                        "licenses": [],
                    },
                )
            )

    def test_policy_schema_validation_unknown_root_key(self, tmp_path):
        with pytest.raises(ValueError, match="unknown root key"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    {
                        "schema_version": 1,
                        "default_decision": "deny",
                        "licenses": [],
                        "extra_key": "value",
                    },
                )
            )

    def test_duplicate_license_identifier_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate license identifier"):
            load_license_policy(
                _policy_path(
                    tmp_path,
                    _policy_with_licenses([_make_allow_dict(), _make_allow_dict()]),
                )
            )
