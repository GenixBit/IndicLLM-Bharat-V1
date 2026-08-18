#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Dataset Approval Validator CLI.

Validates data governance approvals against source dataset manifests for legal compliance,
PII sanitization review, contamination clearance, and authorization status.

Usage:
  python scripts/validate_dataset_approval.py \
    --manifest data/governed/sangraha_hi/manifest.json \
    --approval data/governed/sangraha_hi/approval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.approval import (
    DatasetApproval,
    validate_approval_for_manifest,
)
from bharat.data.manifest import DatasetManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a dataset approval against a dataset manifest"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--approval", required=True, help="Path to approval JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    approval_path = Path(args.approval)

    errors: list[str] = []

    if not manifest_path.exists():
        errors.append(f"Manifest file not found: {manifest_path}")

    if not approval_path.exists():
        errors.append(f"Approval file not found: {approval_path}")

    manifest: DatasetManifest | None = None
    approval: DatasetApproval | None = None

    if not errors:
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            manifest = DatasetManifest.from_dict(data)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in manifest: {e}")
        except ValueError as e:
            errors.append(str(e))

    if not errors:
        try:
            raw = approval_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            approval = DatasetApproval(**data)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in approval: {e}")
        except (ValueError, TypeError) as e:
            errors.append(str(e))

    if manifest is not None and approval is not None:
        issues = validate_approval_for_manifest(approval, manifest)
        if issues:
            errors.extend(issues)

    if args.json:
        result: dict[str, object] = {
            "status": "valid" if not errors else "invalid",
            "manifest_path": str(manifest_path),
            "approval_path": str(approval_path),
            "errors": errors,
        }
        if approval is not None:
            result["approval_id"] = approval.approval_id
            result["dataset_id"] = approval.dataset_id
            result["approver"] = approval.approver
            result["approval_status"] = approval.approval_status
        print(json.dumps(result, indent=2))
    else:
        print("=" * 60)
        print("  🇮🇳 Dataset Approval Validation")
        print("=" * 60)
        if errors:
            print(f"  Status: INVALID ({len(errors)} errors)")
            for e in errors:
                print(f"    - {e}")
            print(
                "error: validation failed:\n" + "\n".join(f"  - {e}" for e in errors),
                file=sys.stderr,
            )
        else:
            print("  Status: VALID")
            if approval:
                print(f"  Approval valid: {approval.approval_id}")
                print(f"  Approval ID: {approval.approval_id}")
                print(f"  Dataset ID : {approval.dataset_id}")
                print(f"  Approver   : {approval.approver}")
                print(f"  Status     : {approval.approval_status.upper()}")
        print("=" * 60)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
