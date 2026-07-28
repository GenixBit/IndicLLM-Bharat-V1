from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import (
    TokenizerAcceptanceThresholds,
    evaluate_tokenizer_acceptance,
)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a local tokenizer evaluation report against deterministic thresholds"
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    if args.output is not None and not args.execute:
        parser.error("--output requires --execute")

    report = _load_json_object(args.report, "evaluation report")
    threshold_payload = _load_json_object(args.thresholds, "threshold configuration")
    if threshold_payload.get("schema_version") != "tokenizer-acceptance-thresholds-v1":
        raise ValueError("unsupported threshold schema_version")
    raw_thresholds = threshold_payload.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("threshold configuration field 'thresholds' must be an object")
    thresholds = TokenizerAcceptanceThresholds.from_dict(raw_thresholds)

    tokenizer_name = args.tokenizer_name
    names = report.get("tokenizer_names")
    if tokenizer_name is None:
        if (
            not isinstance(names, list)
            or len(names) != 1
            or not isinstance(names[0], str)
        ):
            raise ValueError(
                "--tokenizer-name is required when the report does not contain "
                "exactly one tokenizer"
            )
        tokenizer_name = names[0]

    result = evaluate_tokenizer_acceptance(report, tokenizer_name, thresholds)
    serialized = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    )

    if args.output is not None:
        try:
            with open(args.output, "x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing file: {args.output}"
            ) from None

    print(serialized, end="")
    sys.stdout.flush()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
