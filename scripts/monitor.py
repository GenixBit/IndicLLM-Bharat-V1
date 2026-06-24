#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — Training & Infrastructure Monitor

Shows real-time status of all running services and training progress.

Usage:
  python scripts/monitor.py                   # Full dashboard
  python scripts/monitor.py --check-api       # Just API health
  python scripts/monitor.py --check-data      # Data pipeline status
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def print_header():
    print("\033[2J\033[H")  # Clear screen
    print("=" * 60)
    print("  🇮🇳 IndicLLM-Bharat-V1 — Monitor Dashboard")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def check_training_log(log_path: str = "train.log"):
    """Parse training log for latest metrics."""
    path = Path(log_path)
    if not path.exists():
        return None

    lines = path.read_text().strip().split("\n")

    # Get last training iter
    train_lines = [l for l in lines if l.startswith("iter ")]
    eval_lines = [l for l in lines if l.startswith("step ")]

    result = {"total_lines": len(lines)}

    if train_lines:
        last = train_lines[-1]
        parts = last.split(",")
        iter_str = parts[0].split(":")[0].replace("iter ", "").strip()
        loss_str = parts[0].split("loss")[1].strip() if "loss" in parts[0] else "?"
        result["last_iter"] = int(iter_str)
        result["last_loss"] = float(loss_str) if loss_str != "?" else None

    if eval_lines:
        last_eval = eval_lines[-1]
        parts = last_eval.replace("step ", "").split(",")
        result["eval_step"] = int(parts[0].split(":")[0].strip())
        if "train loss" in last_eval:
            result["train_loss"] = float(last_eval.split("train loss")[1].split(",")[0].strip())
        if "val loss" in last_eval:
            result["val_loss"] = float(last_eval.split("val loss")[1].strip())

    return result


def check_checkpoints(ckpt_dir: str = "checkpoints"):
    """List all checkpoints with sizes."""
    path = Path(ckpt_dir)
    if not path.exists():
        return []

    checkpoints = []
    for f in path.rglob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        checkpoints.append({
            "path": str(f),
            "size_mb": round(size_mb, 1),
            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
        })
    return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)


def check_data_stats(data_dir: str = "data"):
    """Check data pipeline status."""
    path = Path(data_dir)
    stats = {}

    # Check shards
    for shard in ["shards/train.bin", "shards/val.bin"]:
        shard_path = path / shard
        if shard_path.exists():
            size_mb = shard_path.stat().st_size / (1024 * 1024)
            stats[shard] = f"{size_mb:.1f} MB"

    # Check Indic data
    indic_path = path / "indic"
    if indic_path.exists():
        txt_files = list(indic_path.rglob("*.txt"))
        total_size = sum(f.stat().st_size for f in txt_files) / (1024 * 1024)
        stats["indic_files"] = len(txt_files)
        stats["indic_size_mb"] = round(total_size, 1)

        # Check per-language
        for lang_dir in sorted(indic_path.iterdir()):
            if lang_dir.is_dir():
                lang_files = list(lang_dir.glob("*.txt"))
                if lang_files:
                    lang_size = sum(f.stat().st_size for f in lang_files) / 1024
                    stats[f"lang_{lang_dir.name}"] = f"{len(lang_files)} files ({lang_size:.0f} KB)"

    # Check train/val bins
    for bin_name in ["train.bin", "val.bin"]:
        bin_path = indic_path / bin_name if indic_path.exists() else None
        if bin_path and bin_path.exists():
            size_mb = bin_path.stat().st_size / (1024 * 1024)
            stats[f"indic_{bin_name}"] = f"{size_mb:.1f} MB"

    return stats


def check_benchmark_results(eval_dir: str = "eval"):
    """Load latest benchmark results."""
    path = Path(eval_dir)
    results = []

    for f in sorted(path.glob("results_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            results.append({
                "file": f.name,
                "val_ppl": data.get("val_perplexity", "?"),
                "val_acc": data.get("val_accuracy", "?"),
                "train_ppl": data.get("train_perplexity", "?"),
                "iter": data.get("iter_num", "?"),
            })
        except Exception:
            pass

    return results


def print_section(title: str, content: dict | list | str):
    """Pretty-print a dashboard section."""
    print(f"\n  {'─' * 40}")
    print(f"  📊 {title}")
    print(f"  {'─' * 40}")

    if isinstance(content, dict):
        for k, v in content.items():
            print(f"    {k:20s} : {v}")
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"    {k:20s} : {v}")
                print()
            else:
                print(f"    {item}")
    else:
        print(f"    {content}")


def main():
    parser = argparse.ArgumentParser(description="IndicLLM Monitor")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    args = parser.parse_args()

    while True:
        print_header()

        # Training
        train = check_training_log(os.path.expanduser("~/train.log"))
        if train:
            print_section("Training Progress", train)

        # Checkpoints
        ckpts = check_checkpoints("checkpoints")
        if ckpts:
            print_section("Checkpoints", ckpts[:3])

        # Data
        data = check_data_stats("data")
        if data:
            print_section("Data Pipeline", data)

        # Benchmarks
        benchmarks = check_benchmark_results("eval")
        if benchmarks:
            print_section("Benchmark Results", benchmarks[:3])

        print(f"\n  {'=' * 40}")
        print(f"  Last updated: {datetime.now().strftime('%H:%M:%S')}")

        if not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
