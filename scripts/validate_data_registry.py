#!/usr/bin/env python3
"""
Validate the Bharat AI data source registry.

Usage:
    python scripts/validate_data_registry.py
    python scripts/validate_data_registry.py --registry-dir data_registry/sources --policy data_registry/license_policy.yaml
    python scripts/validate_data_registry.py --json
    python scripts/validate_data_registry.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.licensing import LicenseDecision, load_license_policy
from bharat.data.registry import DataRegistry
from bharat.data.schema import SourceStatus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_DIR = ROOT / "data_registry" / "sources"
DEFAULT_POLICY_PATH = ROOT / "data_registry" / "license_policy.yaml"


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
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

    if not registry_dir.is_dir():
        _fail(f"registry directory not found: {registry_dir}")

    if not policy_path.exists():
        policy_path = registry_dir / "license_policy.yaml"

    # Load policy first
    try:
        policy = load_license_policy(policy_path)
    except Exception as e:
        _fail(str(e))

    errors: list[str] = []
    warnings: list[str] = []

    registry: DataRegistry | None = None
    try:
        registry = DataRegistry.load(
            registry_dir=registry_dir,
            policy_path=policy_path,
        )
    except Exception as e:
        _fail(str(e))

    assert registry is not None

    sources = registry.list_all()

    if not sources:
        if args.json:
            result = {
                "status": "empty",
                "message": "Registry is structurally valid but contains no sources.",
                "total_records": 0,
                "proposed_count": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "deprecated_count": 0,
                "license_decisions": {},
                "digest": registry.digest(),
                "errors": [],
                "warnings": [],
            }
            print(json.dumps(result, indent=2))
        else:
            print("Registry is structurally valid but contains no sources.")
            print(f"Registry digest: {registry.digest()}")
        sys.exit(0)

    proposed = [s for s in sources if s.status == SourceStatus.PROPOSED]
    approved = [s for s in sources if s.status == SourceStatus.APPROVED]
    rejected = [s for s in sources if s.status == SourceStatus.REJECTED]
    deprecated = [s for s in sources if s.status == SourceStatus.DEPRECATED]

    licence_decisions: dict[str, int] = {}
    for s in sources:
        dec = policy.decision_for(s.license)
        licence_decisions[dec.value] = licence_decisions.get(dec.value, 0) + 1

    # Check approved sources have correct licensing
    for s in approved:
        decision = policy.decision_for(s.license)
        if decision == LicenseDecision.REVIEW:
            errors.append(
                f"Source '{s.source_id}' v{s.version}: approved but licence "
                f"'{s.license}' requires review"
            )
        elif decision == LicenseDecision.DENY:
            errors.append(
                f"Source '{s.source_id}' v{s.version}: approved but licence '{s.license}' is denied"
            )

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.strict and proposed:
        _fail(f"strict mode: {len(proposed)} proposed source(s) remain")

    if args.json:
        result = {
            "status": "valid",
            "total_records": len(sources),
            "proposed_count": len(proposed),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "deprecated_count": len(deprecated),
            "license_decisions": licence_decisions,
            "digest": registry.digest(),
            "errors": errors,
            "warnings": warnings,
        }
        print(json.dumps(result, indent=2))
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
        print(f"Registry digest: {registry.digest()}")
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


if __name__ == "__main__":
    main()
