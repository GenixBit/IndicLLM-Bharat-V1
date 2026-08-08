#!/usr/bin/env python3
"""Verify a local tokenizer evaluation report against a local threshold config.

This CLI is intentionally filesystem-only. It never downloads data, calls a
network service, trains a tokenizer, or writes an artifact. The JSON result is
printed to stdout so callers can capture it themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import ThresholdConfiguration, evaluate_tokenizer_acceptance


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a local tokenizer evaluation report against thresholds."
    )
    parser.add_argument("--report", required=True, type=Path, help="Local evaluation report JSON")
    parser.add_argument("--thresholds", required=True, type=Path, help="Local threshold configuration JSON")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name present in the report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _load_json(args.report)
    threshold_payload = _load_json(args.thresholds)
    config = ThresholdConfiguration.from_payload(threshold_payload)
    result = evaluate_tokenizer_acceptance(report, args.tokenizer, config)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
