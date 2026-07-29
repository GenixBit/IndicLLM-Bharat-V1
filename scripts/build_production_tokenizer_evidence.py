from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bharat.tokenizer.production_evidence_builder import write_candidate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validated local candidate tokenizer evidence manifest.",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--repository-commit-sha", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--evaluation-input", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--acceptance-decision", type=Path, required=True)
    parser.add_argument("--threshold-configuration", type=Path, required=True)
    parser.add_argument("--generating-command", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        digest = write_candidate_manifest(
            args.output,
            evidence_root=args.evidence_root,
            repository_commit_sha=args.repository_commit_sha,
            tokenizer_path=args.tokenizer,
            evaluation_input_path=args.evaluation_input,
            evaluation_report_path=args.evaluation_report,
            acceptance_decision_path=args.acceptance_decision,
            threshold_configuration_path=args.threshold_configuration,
            generating_commands=args.generating_command,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {"manifest_sha256": digest, "output": str(args.output.resolve())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
