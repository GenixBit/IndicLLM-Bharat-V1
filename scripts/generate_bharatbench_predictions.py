#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.eval.adapters import build_prediction_adapter
from bharat.eval.prediction_runner import PredictionRunner, write_predictions_jsonl
from bharat.eval.schema import EvalExample

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {path}:{line_num}: {e}") from e
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid JSONL record at {path}:{line_num}: expected object"
            )
        records.append(record)
    return records


def _load_examples(path: Path) -> list[EvalExample]:
    records = _load_jsonl(path)
    examples: list[EvalExample] = []
    seen_ids: set[str] = set()
    for record in records:
        example = EvalExample.from_dict(record)
        if example.example_id in seen_ids:
            raise ValueError(f"Duplicate example_id {example.example_id!r} in {path}")
        seen_ids.add(example.example_id)
        examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BharatBench prediction JSONL files from local examples"
    )
    parser.add_argument("--examples", required=True, help="Path to examples JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path")
    parser.add_argument(
        "--adapter",
        required=True,
        choices=("expected", "echo", "choice-baseline"),
        help="Deterministic local prediction adapter",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    examples_path = Path(args.examples)
    output_path = Path(args.output)

    if _is_remote_url(str(examples_path)):
        print(f"error: Remote examples path rejected: {examples_path}", file=sys.stderr)
        sys.exit(1)
    if _is_remote_url(str(output_path)):
        print(f"error: Remote output path rejected: {output_path}", file=sys.stderr)
        sys.exit(1)

    if not examples_path.exists():
        print(f"error: Examples file not found: {examples_path}", file=sys.stderr)
        sys.exit(1)

    try:
        examples = _load_examples(examples_path)
        adapter = build_prediction_adapter(args.adapter)
        predictions = PredictionRunner().run(examples, adapter)
        write_predictions_jsonl(predictions, output_path)
    except (TypeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "adapter": args.adapter,
        "examples": len(examples),
        "predictions": len(predictions),
        "output": str(output_path),
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Generated {len(predictions)} predictions using adapter={args.adapter}")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
