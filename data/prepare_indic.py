#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — Indic Language Data Pipeline

Downloads and processes multilingual Indic text data from:
  - AI4Bharat IndicCorp v2 (HuggingFace: ai4bharat/IndicCorp)
  - Sangraha (ai4bharat/sangraha) — curated Indic web text
  - CC-100 Indic subsets (statmt/cc100)
  - ROOTS / mC4 Indic subsets (as fallback)

Produces:
  data/indic/train.bin  — tokenised training shards
  data/indic/val.bin    — tokenised val shard
  data/indic/meta.pkl   — tokenizer metadata
  data/indic/DATASET.md — dataset card

Usage:
  # Quick test (1k docs per language):
  python data/prepare_indic.py --max-docs 1000

  # Full pipeline (100k docs per language):
  python data/prepare_indic.py --max-docs 100000

  # Specific languages only:
  python data/prepare_indic.py --langs hi,bn,ta --max-docs 50000

Supported languages:
  hi=Hindi, bn=Bengali, ta=Tamil, te=Telugu, mr=Marathi,
  gu=Gujarati, kn=Kannada, ml=Malayalam, pa=Punjabi, or=Odia,
  ur=Urdu, as=Assamese, sa=Sanskrit
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Language config ───────────────────────────────────────────
INDIC_LANGS = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "as": "Assamese",
    "sa": "Sanskrit",
}

# Script ranges for quality filtering
INDIC_UNICODE_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "mr": (0x0900, 0x097F),
    "sa": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),  # Bengali
    "as": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "or": (0x0B00, 0x0B7F),  # Oriya
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "ur": (0x0600, 0x06FF),  # Arabic (Urdu uses Arabic script)
}


# ── Quality filter ────────────────────────────────────────────
def is_quality_indic(
    text: str, lang: str, min_chars: int = 50, min_script_ratio: float = 0.3
) -> bool:
    """Return True if text passes basic quality checks."""
    text = text.strip()
    if len(text) < min_chars:
        return False
    if len(text) > 100_000:  # skip huge docs
        return False
    # Check script ratio
    if lang in INDIC_UNICODE_RANGES:
        lo, hi = INDIC_UNICODE_RANGES[lang]
        script_chars = sum(1 for c in text if lo <= ord(c) <= hi)
        if script_chars / max(len(text), 1) < min_script_ratio:
            return False
    # Skip docs with too many URLs or special chars
    url_count = text.count("http")
    if url_count > 5:
        return False
    return True


# ── Data sources ──────────────────────────────────────────────
def stream_sangraha(lang: str, max_docs: int) -> Iterator[str]:
    """Stream from ai4bharat/sangraha — curated, high-quality Indic web text."""
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Loading Sangraha ({INDIC_LANGS.get(lang, lang)})...")
        # Sangraha has three splits: verified, unverified, synthetic
        for subset in ["verified", "unverified"]:
            try:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    subset,
                    split="train",
                    streaming=True,
                )
                count = 0
                for item in ds:
                    if count >= max_docs:
                        break
                    # sangraha has 'text' and 'lang' fields
                    if item.get("lang", lang) != lang:
                        continue
                    text = item.get("text", "")
                    if is_quality_indic(text, lang):
                        yield text
                        count += 1
                if count > 0:
                    print(f"  [{lang}] Sangraha/{subset}: {count} docs")
                    return
            except Exception:
                continue
        print(f"  [{lang}] Sangraha unavailable, trying Wikipedia...")
        yield from stream_wikipedia(lang, max_docs)
    except Exception as e:
        print(f"  [{lang}] Sangraha error ({e}), trying Wikipedia...")
        yield from stream_wikipedia(lang, max_docs)


def stream_wikipedia(lang: str, max_docs: int) -> Iterator[str]:
    """Wikipedia via wikimedia/wikipedia — Parquet format, no auth needed."""
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Loading Wikipedia ({INDIC_LANGS.get(lang, lang)})...")
        ds = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
        count = 0
        for item in ds:
            if count >= max_docs:
                break
            text = item.get("text", "")
            if is_quality_indic(text, lang, min_chars=100):
                yield text
                count += 1
        print(f"  [{lang}] Wikipedia: {count} docs")
    except Exception as e:
        print(f"  [{lang}] Wikipedia unavailable ({e})")


def stream_cc100(lang: str, max_docs: int) -> Iterator[str]:
    """CulturaX — modern multilingual corpus with Parquet format (replaces CC-100)."""
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Loading CulturaX ({INDIC_LANGS.get(lang, lang)})...")
        ds = load_dataset(
            "uonlp/CulturaX",
            lang,
            split="train",
            streaming=True,
        )
        count = 0
        for item in ds:
            if count >= max_docs:
                break
            text = item.get("text", "")
            if is_quality_indic(text, lang):
                yield text
                count += 1
        print(f"  [{lang}] CulturaX: {count} docs")
    except Exception as e:
        print(f"  [{lang}] CulturaX unavailable ({e}) — trying mC4...")
        yield from stream_mc4(lang, max_docs)


def stream_mc4(lang: str, max_docs: int) -> Iterator[str]:
    """mC4 — multilingual C4, widely available on HF."""
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Loading mC4 ({INDIC_LANGS.get(lang, lang)})...")
        ds = load_dataset(
            "allenai/c4",
            "multilingual",
            split="train",
            streaming=True,
        )
        count = 0
        for item in ds:
            if count >= max_docs:
                break
            if item.get("language") != lang:
                continue
            text = item.get("text", "")
            if is_quality_indic(text, lang):
                yield text
                count += 1
        print(f"  [{lang}] mC4: {count} docs")
    except Exception as e:
        print(f"  [{lang}] mC4 unavailable ({e})")


def stream_indiccorp(lang: str, max_docs: int) -> Iterator[str]:
    """IndicCorp v2 — largest Indic corpus."""
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Loading IndicCorp v2...")
        ds = load_dataset(
            "ai4bharat/IndicCorp",
            lang,
            split="train",
            streaming=True,
        )
        count = 0
        for item in ds:
            if count >= max_docs:
                break
            text = item.get("text", item.get("sentence", ""))
            if is_quality_indic(text, lang):
                yield text
                count += 1
        print(f"  [{lang}] IndicCorp: {count} docs")
    except Exception as e:
        print(f"  [{lang}] IndicCorp unavailable: {e}")


# ── Tokenizer ─────────────────────────────────────────────────
def get_or_train_tokenizer(texts_sample: list[str], out_dir: Path, vocab_size: int = 32000):
    """
    Load existing tokenizer or train a new SentencePiece tokenizer
    on a sample of the Indic text corpus.
    """
    tok_path = out_dir / "tokenizer.model"
    if tok_path.exists():
        print(f"  Loading existing tokenizer from {tok_path}")
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))
        return sp

    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit("Install sentencepiece: pip install sentencepiece")

    print(f"  Training SentencePiece tokenizer (vocab={vocab_size})...")
    sample_path = out_dir / "tokenizer_sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for text in texts_sample[:100_000]:
            f.write(text.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.train(
        input=str(sample_path),
        model_prefix=str(out_dir / "tokenizer"),
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        byte_fallback=True,  # Handle OOV chars via byte encoding
        split_digits=True,
        num_threads=os.cpu_count(),
    )
    sample_path.unlink()
    print(f"  Tokenizer saved → {tok_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tok_path))
    return sp


# ── Encoding + shard writing ──────────────────────────────────
def encode_and_write(texts: list[str], tokenizer, out_path: Path, use_gpt2: bool = False) -> int:
    """Tokenize texts and write as uint16 binary shard."""
    all_ids: list[int] = []

    if use_gpt2:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        for text in texts:
            ids = tok.encode(text)
            all_ids.extend(ids)
            all_ids.append(tok.eos_token_id)
    else:
        for text in texts:
            ids = tokenizer.Encode(text, out_type=int)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_id())

    arr = np.array(all_ids, dtype=np.uint16)
    with open(out_path, "wb") as f:
        f.write(arr.tobytes())

    print(f"  Wrote {out_path} ({len(arr):,} tokens)")
    return len(arr)


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IndicLLM Indic data pipeline")
    parser.add_argument(
        "--langs",
        default="hi,bn,ta,te,mr",
        help="Comma-separated language codes (default: hi,bn,ta,te,mr)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=10000, help="Max docs per language (default: 10000)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/indic"),
        help="Output directory (default: data/indic)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000, help="SentencePiece vocab size (default: 32000)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.005, help="Validation split ratio (default: 0.005)"
    )
    parser.add_argument(
        "--source",
        default="sangraha",
        choices=["sangraha", "cc100", "indiccorp", "wikipedia"],
        help="Data source (default: sangraha)",
    )
    parser.add_argument(
        "--use-gpt2-tok",
        action="store_true",
        help="Use GPT-2 tokenizer instead of training SentencePiece",
    )
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  IndicLLM-Bharat — Indic Data Pipeline")
    print(f"  Languages : {', '.join(INDIC_LANGS.get(l, l) for l in langs)}")
    print(f"  Max docs  : {args.max_docs:,} per language")
    print(f"  Source    : {args.source}")
    print(f"  Out dir   : {out_dir}")
    print(f"{'='*60}\n")

    # ── Collect all texts ────────────────────────────────────
    all_texts: list[str] = []
    lang_stats: dict[str, int] = {}

    for lang in langs:
        if lang not in INDIC_LANGS:
            print(f"  WARNING: Unknown language code '{lang}', skipping")
            continue

        if args.source == "sangraha":
            texts = list(stream_sangraha(lang, args.max_docs))
            if not texts:  # auto-fallback to Wikipedia
                texts = list(stream_wikipedia(lang, args.max_docs))
        elif args.source == "cc100":
            texts = list(stream_cc100(lang, args.max_docs))
            if not texts:
                texts = list(stream_wikipedia(lang, args.max_docs))
        elif args.source == "wikipedia":
            texts = list(stream_wikipedia(lang, args.max_docs))
        else:
            texts = list(stream_indiccorp(lang, args.max_docs))

        lang_stats[lang] = len(texts)
        all_texts.extend(texts)
        print(f"  [{lang}] {INDIC_LANGS[lang]}: {len(texts):,} docs collected\n")

    if not all_texts:
        print("ERROR: No texts collected. Check your internet connection and HuggingFace access.")
        sys.exit(1)

    print(f"Total: {len(all_texts):,} docs across {len(langs)} languages\n")

    # Shuffle
    import random

    random.seed(42)
    random.shuffle(all_texts)

    # ── Train tokenizer ──────────────────────────────────────
    if args.use_gpt2_tok:
        print("Using GPT-2 tokenizer (skipping SentencePiece training)")
        tokenizer = None
        vocab_size = 50257
    else:
        tokenizer = get_or_train_tokenizer(all_texts, out_dir, args.vocab_size)
        vocab_size = args.vocab_size

    # ── Train/val split ──────────────────────────────────────
    val_n = max(1, int(len(all_texts) * args.val_ratio))
    val_texts = all_texts[:val_n]
    train_texts = all_texts[val_n:]
    print(f"\nTrain: {len(train_texts):,} docs | Val: {len(val_texts):,} docs")

    # ── Write binary shards ──────────────────────────────────
    train_tokens = encode_and_write(
        train_texts, tokenizer, out_dir / "train.bin", use_gpt2=args.use_gpt2_tok
    )
    val_tokens = encode_and_write(
        val_texts, tokenizer, out_dir / "val.bin", use_gpt2=args.use_gpt2_tok
    )

    # ── Meta pickle ──────────────────────────────────────────
    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "languages": lang_stats,
        "source": args.source,
        "tokenizer": "sentencepiece" if not args.use_gpt2_tok else "gpt2",
        "tokenizer_path": str(out_dir / "tokenizer.model") if not args.use_gpt2_tok else "gpt2",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # ── Dataset card ─────────────────────────────────────────
    card = f"""# IndicLLM-Bharat — Indic Language Dataset

## Summary
- **Source**: {args.source}
- **Languages**: {', '.join(f"{INDIC_LANGS[l]} ({l})" for l in langs if l in INDIC_LANGS)}
- **Train tokens**: {train_tokens:,}
- **Val tokens**: {val_tokens:,}
- **Vocab size**: {vocab_size:,}
- **Tokenizer**: {"SentencePiece BPE" if not args.use_gpt2_tok else "GPT-2"}

## Language Breakdown
| Language | Code | Docs |
|----------|------|------|
"""
    for lang, count in lang_stats.items():
        card += f"| {INDIC_LANGS.get(lang, lang)} | {lang} | {count:,} |\n"

    card += f"""
## Files
- `train.bin` — {train_tokens:,} tokens (uint16)
- `val.bin`   — {val_tokens:,} tokens (uint16)
- `meta.pkl`  — tokenizer metadata
- `tokenizer.model` — SentencePiece model

## Usage
```python
# In configs/gpt2-124m-indic.yaml:
data:
  train_bin: data/indic/train.bin
  val_bin:   data/indic/val.bin
  meta_pkl:  data/indic/meta.pkl
```
"""
    (out_dir / "DATASET.md").write_text(card)
    print(f"\n  Dataset card → {out_dir}/DATASET.md")

    print(f"\n{'='*60}")
    print("  ✅ Indic data pipeline complete!")
    print(f"  Train : {train_tokens:,} tokens")
    print(f"  Val   : {val_tokens:,} tokens")
    print(f"  Output: {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
