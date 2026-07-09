#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.licensing import LicenseDecision, _validate_allow_record, load_license_policy
from bharat.data.registry import DataRegistry
from bharat.data.schema import SourceStatus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_DIR = ROOT / "data_registry" / "sources"
DEFAULT_POLICY_PATH = ROOT / "data_registry" / "license_policy.yaml"


def _fail(msg: str) -> None:
    print(
        json.dumps({"error": msg}) if hasattr(sys, "_json_mode") else f"error: {msg}",
        file=sys.stderr if not hasattr(sys, "_json_mode") else sys.stdout,
    )
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Bharat AI data source registry")
    parser.add_argument(
        "--registry-dir",
        default=str(DEFAULT_REGISTRY_DIR),
        help="Path to source YAML directory (default: data_registry/sources)",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Path to license policy YAML (default: data_registry/license_policy.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any source remains proposed or under review",
    )

    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)
    policy_path = Path(args.policy) if args.policy else DEFAULT_POLICY_PATH

    errors: list[str] = []
    warnings: list[str] = []

    if not registry_dir.is_dir():
        errors.append(f"registry directory not found: {registry_dir}")

    if not policy_path.exists():
        policy_path = registry_dir / "license_policy.yaml"

    policy = None
    registry: DataRegistry | None = None

    if not errors:
        try:
            policy = load_license_policy(policy_path)
        except Exception as e:
            errors.append(str(e))

    if not errors and policy is not None:
        try:
            registry = DataRegistry.load(
                registry_dir=registry_dir,
                policy_path=policy_path,
            )
        except Exception as e:
            errors.append(str(e))

    if registry is None and not errors:
        errors.append("failed to load registry")

    if errors:
        if args.json:
            result = {
                "status": "invalid",
                "errors": errors,
                "warnings": warnings,
            }
            print(json.dumps(result, indent=2))
        else:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    assert registry is not None

    sources = registry.list_all()
    snapshot = registry.to_snapshot()

    proposed = [s for s in sources if s.status == SourceStatus.PROPOSED]
    approved = [s for s in sources if s.status == SourceStatus.APPROVED]
    rejected = [s for s in sources if s.status == SourceStatus.REJECTED]
    deprecated = [s for s in sources if s.status == SourceStatus.DEPRECATED]

    licence_decisions: dict[str, int] = {}
    for s in sources:
        dec = registry.policy.decision_for(s.license)
        licence_decisions[dec.value] = licence_decisions.get(dec.value, 0) + 1

    # Check approved sources have correct licensing
    for s in approved:
        lic = registry.policy.resolve(s.license)
        if lic is None:
            errors.append(
                f"Source '{s.source_id}' v{s.version}: approved but licence "
                f"'{s.license}' not found in policy"
            )
        elif lic.decision == LicenseDecision.REVIEW:
            errors.append(
                f"Source '{s.source_id}' v{s.version}: approved but licence "
                f"'{s.license}' requires review"
            )
        elif lic.decision == LicenseDecision.DENY:
            errors.append(
                f"Source '{s.source_id}' v{s.version}: approved but licence '{s.license}' is denied"
            )
        elif lic.decision == LicenseDecision.ALLOW:
            try:
                _validate_allow_record(lic, f"policy.{lic.identifier}")
            except ValueError as e:
                errors.append(f"Source '{s.source_id}' v{s.version}: {e}")

    # Strict mode checks
    if args.strict:
        if proposed:
            errors.append(f"strict mode: {len(proposed)} proposed source(s) remain")
        for s in sources:
            dec = registry.policy.decision_for(s.license)
            if dec == LicenseDecision.REVIEW:
                errors.append(
                    f"strict mode: source '{s.source_id}' v{s.version} has REVIEW licence "
                    f"'{s.license}'"
                )
            if dec == LicenseDecision.DENY and s.status == SourceStatus.APPROVED:
                errors.append(
                    f"strict mode: source '{s.source_id}' v{s.version} is approved with DENY licence"
                )
        for s in approved:
            lic = registry.policy.resolve(s.license)
            if lic is not None and lic.decision == LicenseDecision.ALLOW:
                try:
                    _validate_allow_record(lic, f"policy.{lic.identifier}")
                except ValueError:
                    errors.append(
                        f"strict mode: source '{s.source_id}' v{s.version}: "
                        f"approved ALLOW licence missing evidence"
                    )
        if warnings:
            errors.extend(f"strict mode: warning: {w}" for w in warnings)

    if args.json:
        result = {
            "status": "invalid" if errors else "valid",
            "total_records": len(sources),
            "proposed_count": len(proposed),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "deprecated_count": len(deprecated),
            "license_decisions": licence_decisions,
            "registry_digest": snapshot["registry_digest"],
            "policy_digest": snapshot["policy_digest"],
            "errors": errors,
            "warnings": warnings,
        }
        if not sources:
            result["status"] = "empty"
        print(json.dumps(result, indent=2))
    else:
        if not sources:
            print("Registry is structurally valid but contains no sources.")
            print(f"Registry digest: {snapshot['registry_digest']}")
            print(f"Policy digest: {snapshot['policy_digest']}")
        else:
            print(f"Total records: {len(sources)}")
            print(f"  Proposed:   {len(proposed)}")
            print(f"  Approved:   {len(approved)}")
            print(f"  Rejected:   {len(rejected)}")
            print(f"  Deprecated: {len(deprecated)}")
            print()
            print("License decisions:")
            for dec, count in sorted(licence_decisions.items()):
                print(f"  {dec}: {count}")
            print()
            print(f"Registry digest: {snapshot['registry_digest']}")
            print(f"Policy digest: {snapshot['policy_digest']}")
            if errors:
                print(f"\nErrors ({len(errors)}):")
                for e in errors:
                    print(f"  - {e}")
            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for w in warnings:
                    print(f"  - {w}")
            if not errors:
                print("Validation passed.")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
