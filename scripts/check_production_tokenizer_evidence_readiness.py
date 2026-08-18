#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Production Tokenizer Evidence Readiness Inspection CLI.

Inspects a local production-tokenizer evidence manifest for review readiness and
human promotion criteria.

Usage:
  python scripts/check_production_tokenizer_evidence_readiness.py evidence_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bharat.tokenizer.production_evidence_readiness import (
    inspect_evidence_readiness,
    write_readiness_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a local production-tokenizer evidence manifest for review readiness."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = inspect_evidence_readiness(args.manifest)
    except Exception as e:
        print(f"error: readiness inspection failed: {e}", file=sys.stderr)
        return 2

    if args.output is None:
        print(json.dumps(report.to_canonical_dict(), sort_keys=True, indent=2))
    else:
        digest = write_readiness_report(args.manifest, args.output)
        print(json.dumps({"output_sha256": digest}, sort_keys=True, indent=2))

    return 0 if report.ready_for_human_promotion else 2


if __name__ == "__main__":
    raise SystemExit(main())
