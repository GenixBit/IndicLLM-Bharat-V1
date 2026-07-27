from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deterministic byte-level BPE tokenizer")
    parser.add_argument("--corpus", required=True, type=Path, help="input corpus file (JSONL)")
    parser.add_argument("--vocab-size", required=True, type=int, help="target vocabulary size")
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="output tokenizer JSON path"
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        default="",
        help=(
            "JSON dict of special token->id, e.g. " '\'{"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}\''
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    special_tokens: dict[str, int] | None = None
    if args.special_tokens:
        try:
            special_tokens = json.loads(args.special_tokens)
        except json.JSONDecodeError as e:
            print(f"error: invalid --special-tokens JSON: {e}", file=sys.stderr)
            sys.exit(1)

    from bharat.tokenizer.bpe import train_bpe

    tokenizer = train_bpe(
        corpus_path=args.corpus,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)

    corpus_sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    print(f"Tokenizer saved to {args.output}")
    print(f"Vocabulary size:  {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"Tokenizer hash:   {tokenizer.tokenizer_hash}")
    print(f"Corpus SHA-256:   {corpus_sha}")


if __name__ == "__main__":
    main()
