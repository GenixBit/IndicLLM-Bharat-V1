#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.data.processing import DataProcessor
from bharat.data.stats import compute_statistics


def _read_texts(path: Path) -> list[str]:
    if path.is_dir():
        texts: list[str] = []
        for f in sorted(path.iterdir()):
            if f.suffix in (".txt", ".jsonl"):
                if f.suffix == ".jsonl":
                    with f.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                texts.append(line)
                else:
                    texts.append(f.read_text(encoding="utf-8"))
        return texts
    elif path.suffix == ".jsonl":
        texts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts
    else:
        return [path.read_text(encoding="utf-8")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics from local text files")
    parser.add_argument("--input", required=True, help="Path to text file, JSONL file, or directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"error: input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    texts = _read_texts(input_path)
    if not texts:
        print("error: no text records found", file=sys.stderr)
        sys.exit(1)

    processor = DataProcessor()
    stats = compute_statistics(texts, processor=processor)

    if args.json:
        print(
            json.dumps(
                {
                    "record_count": stats.record_count,
                    "total_chars": stats.total_chars,
                    "total_utf8_bytes": stats.total_utf8_bytes,
                    "avg_chars": round(stats.avg_chars, 2),
                    "avg_words": round(stats.avg_words, 2),
                    "language_distribution": stats.language_distribution,
                    "quality_score_distribution": stats.quality_score_distribution,
                    "pii_rejection_count": stats.pii_rejection_count,
                    "safety_rejection_count": stats.safety_rejection_count,
                    "duplicate_rejection_count": stats.duplicate_rejection_count,
                    "accepted_count": stats.accepted_count,
                    "rejected_count": stats.rejected_count,
                },
                indent=2,
            )
        )
    else:
        print(f"Dataset Statistics")
        print(f"  Records:              {stats.record_count}")
        print(f"  Total chars:          {stats.total_chars}")
        print(f"  Total UTF-8 bytes:    {stats.total_utf8_bytes}")
        print(f"  Avg chars/record:     {stats.avg_chars:.2f}")
        print(f"  Avg words/record:     {stats.avg_words:.2f}")
        print(f"  Languages:            {dict(sorted(stats.language_distribution.items()))}")
        print(f"  Quality scores:       {dict(sorted(stats.quality_score_distribution.items()))}")
        print(f"  PII rejections:       {stats.pii_rejection_count}")
        print(f"  Safety rejections:    {stats.safety_rejection_count}")
        print(f"  Duplicate rejections: {stats.duplicate_rejection_count}")
        print(f"  Accepted:             {stats.accepted_count}")
        print(f"  Rejected:             {stats.rejected_count}")


if __name__ == "__main__":
    main()
