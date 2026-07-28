from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import (
    TokenizerAcceptanceThresholds,
    evaluate_tokenizer_acceptance,
)

_EXIT_SUCCESS = 0
_EXIT_THRESHOLD_FAILURE = 2
_EXIT_VALIDATION_ERROR = 3


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
        if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
            raise ValueError(
                "--tokenizer-name is required when the report does not contain "
                "exactly one tokenizer"
            )
        tokenizer_name = names[0]

    result = evaluate_tokenizer_acceptance(report, tokenizer_name, thresholds)

    result_json = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )

    written_path: Path | None = None

    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {args.output}")

        expected_digest = result.get("acceptance_sha256", "")
        intermediate_path = args.output.with_name(f".{args.output.name}.tmp")
        try:
            intermediate_path.write_text(result_json, encoding="utf-8")
            written = json.loads(intermediate_path.read_text(encoding="utf-8"))
            written_digest = written.get("acceptance_sha256", "")
            if written_digest != expected_digest:
                msg = (
                    f"acceptance digest mismatch during verification: "
                    f"expected {expected_digest}, got {written_digest}"
                )
                raise RuntimeError(msg)

            written_bytes = intermediate_path.read_bytes()
            final_digest = hashlib.sha256(written_bytes).hexdigest()
            _ = final_digest

            with open(args.output, "xb") as f:
                f.write(written_bytes)
                f.flush()
                os.fsync(f.fileno())
            written_path = args.output

            reread = args.output.read_bytes()
            if reread != written_bytes:
                msg = f"byte-verification failed after final write: {args.output}"
                raise RuntimeError(msg)
        except BaseException:
            if intermediate_path.exists():
                intermediate_path.unlink()
            if written_path is not None and written_path.exists():
                written_path.unlink()
            raise
        finally:
            if intermediate_path.exists():
                intermediate_path.unlink()

    if not args.execute:
        print(result_json, end="")
        sys.stdout.flush()

    if result["passed"]:
        return _EXIT_SUCCESS
    return _EXIT_THRESHOLD_FAILURE


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_VALIDATION_ERROR) from None
