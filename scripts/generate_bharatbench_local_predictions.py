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

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {path}:{line_num}: {e}") from e
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record at {path}:{line_num}: expected object")
        records.append(record)
    return records


def _load_examples(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    seen_ids: set[str] = set()
    for record in _load_jsonl(path):
        example = EvalExample.from_dict(record)
        if example.example_id in seen_ids:
            raise ValueError(f"Duplicate example_id {example.example_id!r} in {path}")
        seen_ids.add(example.example_id)
        examples.append(example)
    return examples


def _reject_remote(name: str, raw_path: str) -> None:
    if _is_remote_url(raw_path):
        raise ValueError(f"Remote {name} path rejected: {raw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BharatBench predictions with a local Bharat checkpoint"
    )
    parser.add_argument("--examples", required=True, help="Path to examples JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSONL path")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local Bharat checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local tokenizer path or directory",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    try:
        _reject_remote("examples", args.examples)
        _reject_remote("output", args.output)
        _reject_remote("checkpoint", args.checkpoint)
        _reject_remote("tokenizer", args.tokenizer)

        examples_path = Path(args.examples)
        output_path = Path(args.output)
        checkpoint_path = Path(args.checkpoint)
        tokenizer_path = Path(args.tokenizer)

        if not examples_path.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_path}")

        config = LocalInferenceConfig(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            do_sample=False,
        )
        examples = _load_examples(examples_path)
        adapter = load_local_causal_lm_adapter(config)
        predictions = PredictionRunner().run(examples, adapter)
        write_predictions_jsonl(predictions, output_path)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "checkpoint": str(checkpoint_path),
        "examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "output": str(output_path),
        "predictions": len(predictions),
        "tokenizer": str(tokenizer_path),
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Generated {len(predictions)} local checkpoint predictions")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
