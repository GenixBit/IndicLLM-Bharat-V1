from __future__ import annotations

import argparse
import json
from pathlib import Path

from bharat.tokenizer.production_evidence_readiness import (
    inspect_evidence_readiness,
    write_readiness_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a local production-tokenizer evidence manifest for review readiness."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_evidence_readiness(args.manifest)
    if args.output is None:
        print(json.dumps(report.to_canonical_dict(), sort_keys=True))
    else:
        digest = write_readiness_report(args.manifest, args.output)
        print(json.dumps({"output_sha256": digest}, sort_keys=True))
    return 0 if report.ready_for_human_promotion else 2


if __name__ == "__main__":
    raise SystemExit(main())
