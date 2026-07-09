#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — Robust Indic Data Downloader

Downloads Indic Wikipedia articles directly from Wikimedia XML dumps
(no HuggingFace account needed, no rate limits).

Also supports AI4Bharat IndicCorp via direct download if HF_TOKEN is set.

Usage:
  # Download top-5 languages, 5000 articles each:
  python data/download_indic.py --langs hi,bn,ta,te,mr --max-articles 5000

  # All 13 Indic languages:
  python data/download_indic.py --langs all --max-articles 10000

  # With HF token (enables Sangraha access):
  HF_TOKEN=hf_xxx python data/download_indic.py --langs hi,bn --use-sangraha

Outputs (same format as prepare_data.py):
  data/indic/train.bin
  data/indic/val.bin
  data/indic/meta.pkl
  data/indic/DATASET.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    raise SystemExit("Install requests: pip install requests")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Wikimedia dump URLs (always public, no auth)
WIKI_DUMP_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-abstract.xml.gz"
WIKI_BZ2_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream-index.txt.bz2"

# Direct article API (no dump needed)
WIKI_API_URL = "https://{lang}.wikipedia.org/w/api.php"

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

INDIC_UNICODE_RANGES = {
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "sa": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "as": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "or": (0x0B00, 0x0B7F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
    "ur": (0x0600, 0x06FF),
}


def is_quality(text: str, lang: str, min_chars: int = 50) -> bool:
    text = text.strip()
    if len(text) < min_chars or len(text) > 200_000:
        return False
    if lang in INDIC_UNICODE_RANGES:
        lo, hi = INDIC_UNICODE_RANGES[lang]
        ratio = sum(1 for c in text if lo <= ord(c) <= hi) / max(len(text), 1)
        if ratio < 0.10:  # 10% script chars — accepts mixed Indic+English articles
            return False
    return True


def fetch_wiki_articles(lang: str, max_articles: int, cache_dir: Path) -> list[str]:
    """
    Fetch Wikipedia articles using generator=random — no title encoding issues,
    works cleanly for all Indic scripts.
    """
    cache_file = cache_dir / f"wiki_{lang}_{max_articles}.jsonl"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f if line.strip()]
        if texts:
            print(f"  [{lang}] Cached: {len(texts)} articles")
            return texts
        cache_file.unlink()

    api_url = WIKI_API_URL.format(lang=lang)
    session = _requests.Session()
    session.headers.update(
        {
            "User-Agent": "IndicLLM-Bharat/1.0 (research)",
            "Accept-Encoding": "identity",  # disable gzip to avoid decode issues
        }
    )

    texts: list[str] = []
    print(f"  [{lang}] Fetching {INDIC_LANGS[lang]} Wikipedia articles via random generator...")

    with open(cache_file, "w", encoding="utf-8") as cache_f:
        attempts = 0
        retry_delay = 2.0
        while len(texts) < max_articles and attempts < max_articles * 3:
            params = {
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": "0",
                "grnlimit": "20",
                "prop": "extracts",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exchars": "5000",
            }
            try:
                resp = session.get(api_url, params=params, timeout=30)
                if resp.status_code == 429:
                    print(f"  [{lang}] Rate limited, sleeping {retry_delay:.0f}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)  # cap at 60s
                    continue
                retry_delay = 2.0  # reset on success
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [{lang}] API error: {e}, retrying...")
                time.sleep(retry_delay)
                attempts += 20
                continue

            for p in data.get("query", {}).get("pages", {}).values():
                text = p.get("extract", "").strip()
                if is_quality(text, lang):
                    texts.append(text)
                    cache_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

            attempts += 20
            time.sleep(1.0)  # polite 1s delay between requests
            if len(texts) % 100 == 0 and len(texts) > 0:
                print(f"  [{lang}] {len(texts)}/{max_articles}...")

    print(f"  [{lang}] {INDIC_LANGS[lang]}: {len(texts)} articles")
    return texts


def fetch_via_hf(lang: str, max_docs: int) -> list[str]:
    """Use HuggingFace datasets with token if available."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return []
    try:
        from datasets import load_dataset

        print(f"  [{lang}] Trying ai4bharat/sangraha with HF token...")
        ds = load_dataset(
            "ai4bharat/sangraha",
            "verified",
            split="train",
            streaming=True,
            token=token,
        )
        texts = []
        for item in ds:
            if len(texts) >= max_docs:
                break
            if item.get("lang") == lang:
                text = item.get("text", "")
                if is_quality(text, lang):
                    texts.append(text)
        print(f"  [{lang}] Sangraha: {len(texts)} docs")
        return texts
    except Exception as e:
        print(f"  [{lang}] HF fetch failed: {e}")
        return []


def tokenize_and_write(texts: list[str], out_path: Path, use_gpt2: bool = True) -> int:
    """Encode texts and write uint16 binary shard."""
    if use_gpt2:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        all_ids: list[int] = []
        for text in texts:
            ids = tok.encode(text, truncation=False)
            all_ids.extend(ids)
            all_ids.append(tok.eos_token_id)
    else:
        import sentencepiece as spm

        # Load trained tokenizer
        sp_path = out_path.parent / "tokenizer.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_path}")
        sp = spm.SentencePieceProcessor()
        sp.Load(str(sp_path))
        all_ids = []
        for text in texts:
            all_ids.extend(sp.Encode(text, out_type=int))
            all_ids.append(sp.eos_id())

    arr = np.array(all_ids, dtype=np.uint16)
    with open(out_path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Wrote {out_path.name}: {len(arr):,} tokens ({len(arr) / 1e6:.1f}M)")
    return len(arr)


def main():
    parser = argparse.ArgumentParser(description="IndicLLM Indic data downloader")
    parser.add_argument("--langs", default="hi,bn,ta,te,mr")
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/indic"))
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument(
        "--use-sangraha",
        action="store_true",
        help="Use Sangraha via HF_TOKEN (overrides Wikipedia)",
    )
    args = parser.parse_args()

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    if args.langs == "all":
        langs = list(INDIC_LANGS.keys())

    out_dir = ROOT / args.out_dir
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat — Indic Data Downloader")
    print(f"  Languages : {', '.join(INDIC_LANGS.get(lang, lang) for lang in langs)}")
    print(f"  Articles  : up to {args.max_articles:,} per language")
    print(f"  Output    : {out_dir}")
    print(f"{'=' * 60}\n")

    all_texts: list[str] = []
    lang_stats: dict[str, int] = {}

    for lang in langs:
        if lang not in INDIC_LANGS:
            print(f"  Skipping unknown lang: {lang}")
            continue

        # Try HF/Sangraha first if token available, else Wikipedia API
        texts = fetch_via_hf(lang, args.max_articles) if args.use_sangraha else []

        if not texts:
            texts = fetch_wiki_articles(lang, args.max_articles, cache_dir)

        lang_stats[lang] = len(texts)
        all_texts.extend(texts)

    if not all_texts:
        print("ERROR: No texts collected.")
        sys.exit(1)

    total = len(all_texts)
    print(f"\nTotal: {total:,} articles across {len(langs)} languages")

    import random

    random.seed(42)
    random.shuffle(all_texts)

    val_n = max(1, int(total * args.val_ratio))
    val_texts = all_texts[:val_n]
    train_texts = all_texts[val_n:]

    train_tokens = tokenize_and_write(train_texts, out_dir / "train.bin")
    val_tokens = tokenize_and_write(val_texts, out_dir / "val.bin")

    meta = {
        "vocab_size": 50257,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "languages": lang_stats,
        "source": "wikimedia-api",
        "tokenizer": "gpt2",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Dataset card
    card = "# IndicLLM-Bharat Indic Dataset\n\n"
    card += f"- **Languages**: {', '.join(INDIC_LANGS.get(lang, lang) for lang in langs)}\n"
    card += f"- **Train tokens**: {train_tokens:,}\n"
    card += f"- **Val tokens**: {val_tokens:,}\n"
    card += "- **Source**: Wikimedia Wikipedia API\n\n"
    card += "| Language | Docs |\n|----------|------|\n"
    for lang, count in lang_stats.items():
        card += f"| {INDIC_LANGS.get(lang, lang)} | {count:,} |\n"
    (out_dir / "DATASET.md").write_text(card)

    print(f"\n{'=' * 60}")
    print("  ✅ Done!")
    print(f"  Train: {train_tokens:,} tokens  ({train_tokens / 1e6:.1f}M)")
    print(f"  Val  : {val_tokens:,} tokens")
    print(f"  Files: {out_dir}/")
    print("\n  Next: python train/pretrain.py --config configs/gpt2-124m-indic.yaml")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
