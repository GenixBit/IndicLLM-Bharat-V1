from __future__ import annotations

import argparse
import json
from pathlib import Path

from bharat.tokenizer.sampler import sample_tokenizer_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a deterministic tokenizer corpus from an approved local dataset release."
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-path", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = sample_tokenizer_corpus(
        args.release_dir,
        args.output_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.manifest_path is None:
        print(payload, end="")
    else:
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
