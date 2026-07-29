from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bharat.tokenizer.production_evidence import validate_production_evidence

_EXIT_ACCEPTED = 0
_EXIT_VALID_CANDIDATE = 1
_EXIT_INVALID = 2
_EXIT_EXISTING_OUTPUT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a caller-provided local production tokenizer evidence package."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _publish(result_bytes: bytes, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(str(output_path))
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    created_by_us = True
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(result_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        reread = output_path.read_bytes()
        if reread != result_bytes:
            output_path.unlink()
            msg = "byte-verification failed after write"
            raise RuntimeError(msg)
    except BaseException:
        if created_by_us and output_path.exists():
            output_path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_production_evidence(args.manifest)
    payload = result.canonical_bytes()
    if args.output is not None:
        try:
            _publish(payload, args.output)
        except FileExistsError:
            print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
            return _EXIT_EXISTING_OUTPUT
    print(payload.decode("utf-8"))
    if result.accepted:
        return _EXIT_ACCEPTED
    if result.valid:
        return _EXIT_VALID_CANDIDATE
    return _EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
