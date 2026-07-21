#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.approval import DatasetApproval, validate_approval_for_manifest
from bharat.data.manifest import DatasetManifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset approval against its manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--approval", required=True, help="Path to approval JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()
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
            "status": "invalid" if errors else "valid",
            "manifest_path": str(manifest_path),
            "approval_path": str(approval_path),
            "errors": errors,
        }
        if manifest is not None:
            result["manifest_dataset_id"] = manifest.dataset_id
        if approval is not None:
            result["approval_dataset_id"] = approval.dataset_id
            result["approval_status"] = approval.approval_status
        print(json.dumps(result))
    else:
        if not errors and approval is not None and manifest is not None:
            status_line = (
                f"Approval {approval.approval_id} for "
                f"{manifest.dataset_id} ({approval.approval_status})"
            )
            print(f"Approval valid: {status_line}")
        else:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
