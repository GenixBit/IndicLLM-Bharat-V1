from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from bharat.tokenizer.acceptance import (
    ThresholdConfiguration,
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


def _publish(
    result_json: str,
    output_path: Path,
    expected_digest: str,
) -> Path:
    """Publish *result_json* to *output_path* with safe publication.

    Writes to a secure temporary path first, verifies the acceptance digest,
    then atomically publishes with exclusive no-overwrite creation, flushes,
    fsyncs, re-reads and byte-verifies.

    Returns the final path.  Cleans up on failure.
    """
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")

    tmp_name = f".{output_path.name}.{secrets.token_hex(8)}.tmp"
    tmp_path = output_path.with_name(tmp_name)

    written_path: Path | None = None

    try:
        tmp_path.write_text(result_json, encoding="utf-8")
        parsed = json.loads(tmp_path.read_text(encoding="utf-8"))
        parsed_digest = parsed.get("acceptance_sha256", "")
        if parsed_digest != expected_digest:
            msg = (
                f"acceptance digest mismatch during verification: "
                f"expected {expected_digest}, got {parsed_digest}"
            )
            raise RuntimeError(msg)

        written_bytes = tmp_path.read_bytes()

        with open(output_path, "xb") as f:
            f.write(written_bytes)
            f.flush()
            os.fsync(f.fileno())
        written_path = output_path

        reread = output_path.read_bytes()
        if reread != written_bytes:
            msg = f"byte-verification failed after final write: {output_path}"
            raise RuntimeError(msg)

        return written_path
    except BaseException:
        if written_path is not None and written_path.exists():
            written_path.unlink()
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    if args.output is not None and not args.execute:
        parser.error("--output requires --execute")

    report = _load_json_object(args.report, "evaluation report")
    threshold_payload = _load_json_object(args.thresholds, "threshold configuration")

    config = ThresholdConfiguration.from_payload(threshold_payload)

    tokenizer_name = args.tokenizer_name
    names = report.get("tokenizer_names")
    if tokenizer_name is None:
        if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
            raise ValueError(
                "--tokenizer-name is required when the report does not contain "
                "exactly one tokenizer"
            )
        tokenizer_name = names[0]

    result = evaluate_tokenizer_acceptance(
        report, tokenizer_name, config.thresholds, threshold_config=config
    )

    result_json = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )

    if args.dry_run:
        print("--- dry-run: validation passed ---")
        print(result_json, end="")
        sys.stdout.flush()
        return _EXIT_SUCCESS

    if args.output is not None:
        expected_digest = result.get("acceptance_sha256", "")
        _publish(result_json, args.output, expected_digest)

    if not args.execute or args.output is None:
        print(result_json, end="")
        sys.stdout.flush()

    if result["passed"]:
        return _EXIT_SUCCESS
    return _EXIT_THRESHOLD_FAILURE


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_VALIDATION_ERROR) from None
