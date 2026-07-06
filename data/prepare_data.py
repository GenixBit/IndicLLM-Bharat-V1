#!/usr/bin/env python3
"""
Download FineWeb-Edu subset, clean, tokenize, and shard into nanoGPT/LitGPT binary format.

Usage:
  python data/prepare_data.py --subset sample-10BT --max-docs 500   # dry run
  python data/prepare_data.py --subset sample-10BT                # full pipeline
  python data/prepare_data.py --train-tokenizer --vocab-size 32000  # custom BPE
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if len(text) < 50:
        return ""
    return text


def download_fineweb_edu(subset: str, max_docs: int | None) -> list[str]:
    from datasets import load_dataset

    print(f"Loading HuggingFaceFW/fineweb-edu subset={subset} ...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True)

    texts: list[str] = []
    for i, row in enumerate(tqdm(ds, desc="Downloading")):
        if max_docs is not None and i >= max_docs:
            break
        text = clean_text(row.get("text", ""))
        if text:
            texts.append(text + "\n")
    print(f"Collected {len(texts)} documents.")
    return texts


def train_custom_tokenizer(texts: list[str], vocab_size: int, out_dir: Path) -> Path:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))
    print(f"Saved custom tokenizer to {tok_path}")
    return tok_path


def get_tokenizer(tokenizer_dir: Path | None, vocab_size: int, texts: list[str]):
    if tokenizer_dir and (tokenizer_dir / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    if tokenizer_dir:
        train_custom_tokenizer(texts[: min(len(texts), 50000)], vocab_size, tokenizer_dir)
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    from transformers import GPT2TokenizerFast

    print("Using GPT-2 tokenizer (vocab_size=50257).")
    return GPT2TokenizerFast.from_pretrained("gpt2")


def encode_texts(tokenizer, texts: list[str]) -> np.ndarray:
    ids: list[int] = []
    eot = getattr(tokenizer, "eos_token_id", None)
    if eot is None:
        eot = tokenizer.token_to_id("<|endoftext|>") or 50256

    for text in tqdm(texts, desc="Tokenizing"):
        if hasattr(tokenizer, "encode"):
            chunk = tokenizer.encode(text, add_special_tokens=False)
        else:
            chunk = tokenizer.encode(text).ids
        ids.extend(chunk)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)


def write_shards(ids: np.ndarray, out_dir: Path, val_fraction: float = 0.01) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    split = int(n * (1 - val_fraction))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    vocab_size = int(ids.max()) + 1
    meta = {"vocab_size": vocab_size, "train_tokens": len(train_ids), "val_tokens": len(val_ids)}
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Wrote {train_path} ({len(train_ids):,} tokens)")
    print(f"Wrote {val_path} ({len(val_ids):,} tokens)")
    print(f"Meta: {meta}")


def write_dataset_card(out_dir: Path, subset: str, num_docs: int, meta: dict) -> None:
    card = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
---

# llm-lab FineWeb-Edu shard

- Source: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Subset: `{subset}`
- Documents processed: {num_docs:,}
- Train tokens: {meta["train_tokens"]:,}
- Val tokens: {meta["val_tokens"]:,}
- Vocab size: {meta["vocab_size"]:,}

## Filters applied

- Whitespace normalization
- Minimum document length: 50 characters
- Train/val split: 99% / 1%
"""
    (out_dir / "DATASET.md").write_text(card)
    print(f"Wrote dataset card to {out_dir / 'DATASET.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu training shards")
    parser.add_argument("--subset", default="sample-10BT", help="FineWeb-Edu subset name")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit docs for dry runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "shards")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "data" / "tokenizer")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train custom BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    args = parser.parse_args()

    texts = download_fineweb_edu(args.subset, args.max_docs)
    if not texts:
        raise SystemExit("No text collected.")

    tok_dir = args.tokenizer_dir if args.train_tokenizer else None
    tokenizer = get_tokenizer(tok_dir, args.vocab_size, texts)
    ids = encode_texts(tokenizer, texts)
    write_shards(ids, args.out_dir, args.val_fraction)

    with open(args.out_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    write_dataset_card(args.out_dir, args.subset, len(texts), meta)


if __name__ == "__main__":
    main()
