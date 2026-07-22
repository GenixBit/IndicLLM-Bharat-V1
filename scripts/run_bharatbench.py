#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.eval.reporting import BharatBenchReport, compute_aggregate_scores
from bharat.eval.runner import BharatBenchRunner
from bharat.eval.schema import EvalExample, EvalPrediction

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


def _load_predictions(path: Path) -> list[EvalPrediction]:
    records = _load_jsonl(path)
    predictions: list[EvalPrediction] = []
    seen_ids: set[str] = set()
    for record in records:
        example_id = record.get("example_id")
        prediction = record.get("prediction")
        if not isinstance(example_id, str):
            raise ValueError("prediction example_id must be a string")
        if not isinstance(prediction, str):
            raise ValueError("prediction must be a string")
        pred = EvalPrediction(example_id=example_id, prediction=prediction)
        if pred.example_id in seen_ids:
            raise ValueError(
                f"Duplicate prediction example_id {pred.example_id!r} in {path}"
            )
        seen_ids.add(pred.example_id)
        predictions.append(pred)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BharatBench evaluation on local prediction files"
    )
    parser.add_argument("--examples", required=True, help="Path to examples JSONL file")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory for report")
    parser.add_argument("--created-at", default="", help="ISO-8601 UTC timestamp")
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    examples_path = Path(args.examples)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    if _is_remote_url(str(examples_path)):
        print(f"error: Remote examples path rejected: {examples_path}", file=sys.stderr)
        sys.exit(1)
    if _is_remote_url(str(predictions_path)):
        print(
            f"error: Remote predictions path rejected: {predictions_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not examples_path.exists():
        print(f"error: Examples file not found: {examples_path}", file=sys.stderr)
        sys.exit(1)
    if not predictions_path.exists():
        print(f"error: Predictions file not found: {predictions_path}", file=sys.stderr)
        sys.exit(1)

    try:
        examples = _load_examples(examples_path)
        predictions = _load_predictions(predictions_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    runner = BharatBenchRunner()
    try:
        results = runner.run(examples, predictions)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    task_counts: dict[str, int] = {}
    for ex in examples:
        task_counts[ex.task_type] = task_counts.get(ex.task_type, 0) + 1

    aggregate_scores = compute_aggregate_scores(results)

    report = BharatBenchReport(
        run_id=f"bharatbench-{args.created_at[:10] if args.created_at else 'unknown'}",
        example_count=len(examples),
        task_counts=task_counts,
        aggregate_scores=aggregate_scores,
        results=tuple(results),
        created_at=args.created_at or "2026-07-20T00:00:00Z",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "bharatbench_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if args.json:
        result = {
            "status": "success",
            "run_id": report.run_id,
            "example_count": report.example_count,
            "task_counts": dict(report.task_counts),
            "aggregate_scores": dict(report.aggregate_scores),
            "report_path": str(report_path),
        }
        print(json.dumps(result))
    else:
        print(f"BharatBench run: {report.run_id}")
        print(f"  Examples:    {report.example_count}")
        print(f"  Tasks:       {dict(report.task_counts)}")
        print(f"  Report:      {report_path}")
        for key, val in sorted(report.aggregate_scores.items()):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
