#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.eval.local_inference import (
    LocalInferenceConfig,
    load_local_causal_lm_adapter,
)
from bharat.eval.prediction_runner import PredictionRunner, write_predictions_jsonl
from bharat.eval.schema import EvalExample

_URL_RE = re.compile(r"^(https?|ftp|s3|gs):/+", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record at {path}:{line_num}: expected object")
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
        description="Generate BharatBench predictions using a local model checkpoint"
    )
    parser.add_argument("--examples", required=True, help="Path to examples JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local path to model checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local path to tokenizer directory or file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model inference (default: cpu)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    if _is_remote_url(args.examples):
        print(
            f"error: Remote examples path rejected: {args.examples}",
            file=sys.stderr,
        )
        sys.exit(1)
    if _is_remote_url(args.output):
        print(
            f"error: Remote output path rejected: {args.output}",
            file=sys.stderr,
        )
        sys.exit(1)
    if _is_remote_url(args.checkpoint):
        print(
            f"error: Remote checkpoint path rejected: {args.checkpoint}",
            file=sys.stderr,
        )
        sys.exit(1)
    if _is_remote_url(args.tokenizer):
        print(
            f"error: Remote tokenizer path rejected: {args.tokenizer}",
            file=sys.stderr,
        )
        sys.exit(1)

    examples_path = Path(args.examples)
    output_path = Path(args.output)

    if not examples_path.exists():
        print(f"error: Examples file not found: {examples_path}", file=sys.stderr)
        sys.exit(1)

    try:
        examples = _load_examples(examples_path)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    config = LocalInferenceConfig(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    try:
        adapter = load_local_causal_lm_adapter(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        predictions = PredictionRunner().run(examples, adapter)
        write_predictions_jsonl(predictions, output_path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    result = {
        "checkpoint": str(config.checkpoint),
        "examples": len(examples),
        "predictions": len(predictions),
        "output": str(output_path),
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"Generated {len(predictions)} predictions "
            f"using checkpoint={config.checkpoint}"
        )
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
