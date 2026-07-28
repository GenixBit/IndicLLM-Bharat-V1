from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bharat.tokenizer.production_evidence import validate_production_evidence


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.write(b"\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a caller-provided local production tokenizer evidence package."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_production_evidence(args.manifest)
    payload = result.canonical_bytes()
    if args.output is not None:
        try:
            _write_exclusive(args.output, payload)
        except FileExistsError:
            print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
            return 2
    print(payload.decode("utf-8"))
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
