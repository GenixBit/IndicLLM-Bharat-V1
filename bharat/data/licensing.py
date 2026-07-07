from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class LicenseDecision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    DENY = "deny"


_HTTPS_URL_RE = re.compile(r"^https://\S+$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2}))?$")


def _validate_allow_record(record: LicenseRecord, prefix: str) -> None:
    """Validate that an ALLOW record has all required evidence fields."""
    errors: list[str] = []
    if not record.evidence_url:
        errors.append("evidence_url is required")
    elif not _HTTPS_URL_RE.match(record.evidence_url):
        errors.append(f"evidence_url must be a valid https:// URL, got '{record.evidence_url}'")
    if not record.verified_at:
        errors.append("verified_at is required")
    elif not _ISO_DATE_RE.match(record.verified_at):
        errors.append(f"verified_at must be a valid ISO-8601 date, got '{record.verified_at}'")
    if not record.verified_by:
        errors.append("verified_by is required")
    if record.commercial_use_allowed is not True:
        errors.append("commercial_use_allowed must be true")
    if record.model_training_allowed is not True:
        errors.append("model_training_allowed must be true")
    if record.redistribution_allowed is not True:
        errors.append("redistribution_allowed must be true")
    if record.attribution_required is None:
        errors.append("attribution_required is required")
    if record.share_alike is None:
        errors.append("share_alike is required")
    if errors:
        raise ValueError(f"{prefix}: ALLOW record missing required fields: {'; '.join(errors)}")


@dataclass(frozen=True)
class LicenseRecord:
    identifier: str
    name: str
    decision: LicenseDecision
    evidence_url: str | None = None
    verified_at: str | None = None
    verified_by: str | None = None
    commercial_use_allowed: bool | None = None
    model_training_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    attribution_required: bool | None = None
    share_alike: bool | None = None
    conditions: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class LicensePolicy:
    schema_version: int
    default_decision: LicenseDecision
    licenses: tuple[LicenseRecord, ...]

    def resolve(self, identifier: str) -> LicenseRecord | None:
        for lic in self.licenses:
            if lic.identifier == identifier:
                return lic
        return None

    def decision_for(self, identifier: str) -> LicenseDecision:
        record = self.resolve(identifier)
        if record is None:
            return LicenseDecision.DENY
        return record.decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "default_decision": self.default_decision.value,
            "licenses": [
                {
                    k: v
                    for k, v in {
                        "identifier": lic.identifier,
                        "name": lic.name,
                        "decision": lic.decision.value,
                        "evidence_url": lic.evidence_url,
                        "verified_at": lic.verified_at,
                        "verified_by": lic.verified_by,
                        "commercial_use_allowed": lic.commercial_use_allowed,
                        "model_training_allowed": lic.model_training_allowed,
                        "redistribution_allowed": lic.redistribution_allowed,
                        "attribution_required": lic.attribution_required,
                        "share_alike": lic.share_alike,
                        "conditions": lic.conditions,
                        "notes": lic.notes,
                    }.items()
                    if v is not None
                }
                for lic in self.licenses
            ],
        }


_VALID_LICENSES_KEYS = frozenset(
    {
        "identifier",
        "name",
        "decision",
        "evidence_url",
        "verified_at",
        "verified_by",
        "commercial_use_allowed",
        "model_training_allowed",
        "redistribution_allowed",
        "attribution_required",
        "share_alike",
        "conditions",
        "notes",
    }
)


def _validate_license_record(data: dict[str, Any], path: str, index: int) -> LicenseRecord:
    prefix = f"{path}.licenses[{index}]"
    unknown = set(data) - _VALID_LICENSES_KEYS
    if unknown:
        raise ValueError(f"{prefix}: unknown key(s): {', '.join(sorted(unknown))}")

    identifier = data.get("identifier", "")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{prefix}: identifier must be a non-empty string")

    name = data.get("name", "")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{prefix}: name must be a non-empty string")

    decision_val = data.get("decision", "")
    if not isinstance(decision_val, str) or not decision_val:
        raise ValueError(f"{prefix}: decision must be a non-empty string")
    try:
        decision = LicenseDecision(decision_val.lower())
    except ValueError:
        raise ValueError(
            f"{prefix}: unknown decision '{decision_val}'; "
            f"valid: {', '.join(d.value for d in LicenseDecision)}"
        )

    evidence_url = data.get("evidence_url")
    verified_at = data.get("verified_at")
    verified_by = data.get("verified_by")

    commercial_use_allowed = data.get("commercial_use_allowed")
    model_training_allowed = data.get("model_training_allowed")
    redistribution_allowed = data.get("redistribution_allowed")
    attribution_required = data.get("attribution_required")
    share_alike = data.get("share_alike")
    conditions = data.get("conditions")
    notes = data.get("notes")

    for field in (
        "commercial_use_allowed",
        "model_training_allowed",
        "redistribution_allowed",
        "attribution_required",
        "share_alike",
    ):
        val = data.get(field)
        if val is not None and not isinstance(val, bool):
            raise TypeError(f"{prefix}: {field} must be a boolean, got {type(val).__name__}")

    for field in ("evidence_url", "verified_at", "verified_by", "conditions", "notes"):
        val = data.get(field)
        if val is not None and not isinstance(val, str):
            raise TypeError(f"{prefix}: {field} must be a string, got {type(val).__name__}")

    record = LicenseRecord(
        identifier=identifier,
        name=name,
        decision=decision,
        evidence_url=evidence_url,
        verified_at=verified_at,
        verified_by=verified_by,
        commercial_use_allowed=commercial_use_allowed,
        model_training_allowed=model_training_allowed,
        redistribution_allowed=redistribution_allowed,
        attribution_required=attribution_required,
        share_alike=share_alike,
        conditions=conditions,
        notes=notes,
    )

    if decision == LicenseDecision.ALLOW:
        _validate_allow_record(record, prefix)

    return record


LICENSE_POLICY_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "default_decision",
        "licenses",
    }
)


def load_license_policy(path: str | Path) -> LicensePolicy:
    path = Path(path)
    file_path = str(path)

    if not path.exists():
        raise FileNotFoundError(f"License policy not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{file_path}: YAML root must be a mapping, got {type(data).__name__}")

    unknown = set(data) - LICENSE_POLICY_ROOT_KEYS
    if unknown:
        raise ValueError(f"{file_path}: unknown root key(s): {', '.join(sorted(unknown))}")

    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError(
            f"{file_path}: schema_version must be an integer, got {type(schema_version).__name__}"
        )
    if schema_version != 1:
        raise ValueError(f"{file_path}: unsupported schema_version {schema_version}, expected 1")

    default_val = data.get("default_decision", "deny")
    if not isinstance(default_val, str):
        raise TypeError(
            f"{file_path}: default_decision must be a string, got {type(default_val).__name__}"
        )
    try:
        default_decision = LicenseDecision(default_val.lower())
    except ValueError:
        raise ValueError(
            f"{file_path}: unknown default_decision '{default_val}'; "
            f"valid: {', '.join(d.value for d in LicenseDecision)}"
        )
    if default_decision != LicenseDecision.DENY:
        raise ValueError(f"{file_path}: default_decision must be 'deny', got '{default_val}'")

    licenses_raw = data.get("licenses", [])
    if not isinstance(licenses_raw, list):
        raise TypeError(f"{file_path}: licenses must be a list, got {type(licenses_raw).__name__}")

    seen_ids: set[str] = set()
    license_records: list[LicenseRecord] = []
    for i, lic_data in enumerate(licenses_raw):
        if not isinstance(lic_data, dict):
            raise TypeError(
                f"{file_path}.licenses[{i}]: must be a mapping, got {type(lic_data).__name__}"
            )
        record = _validate_license_record(lic_data, file_path, i)
        if record.identifier in seen_ids:
            raise ValueError(
                f"{file_path}.licenses[{i}]: duplicate license identifier '{record.identifier}'"
            )
        seen_ids.add(record.identifier)
        license_records.append(record)

    return LicensePolicy(
        schema_version=schema_version,
        default_decision=default_decision,
        licenses=tuple(license_records),
    )
