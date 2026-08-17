#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Training, Infrastructure & Telemetry Monitor.

Real-time monitoring dashboard and telemetry reporter for IndicLLM-Bharat.
Inspects hardware accelerators (CUDA/MPS/CPU), active checkpoints, governed datasets,
and evaluation benchmark runs.

Usage:
  # Print single-shot dashboard
  python scripts/monitor.py

  # Output JSON status snapshot
  python scripts/monitor.py --json

  # Live watch mode with 5s refresh
  python scripts/monitor.py --watch --interval 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def get_hardware_telemetry() -> dict[str, Any]:
    """Inspect local hardware acceleration and memory."""
    telemetry: dict[str, Any] = {
        "platform": sys.platform,
        "cpu_count": os.cpu_count() or 1,
        "device": "cpu",
        "accelerator_available": False,
    }

    if torch.cuda.is_available():
        telemetry["device"] = "cuda"
        telemetry["accelerator_available"] = True
        telemetry["cuda_device_count"] = torch.cuda.device_count()
        telemetry["cuda_device_name"] = torch.cuda.get_device_name(0)
        telemetry["cuda_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / (1024 * 1024), 1
        )
        telemetry["cuda_memory_reserved_mb"] = round(
            torch.cuda.memory_reserved(0) / (1024 * 1024), 1
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        telemetry["device"] = "mps"
        telemetry["accelerator_available"] = True
        telemetry["mps_backend"] = "Apple Silicon MPS (Unified Memory)"
    else:
        telemetry["device"] = "cpu"
        telemetry["accelerator_available"] = False

    return telemetry


def check_checkpoints(ckpt_dir: str | Path = "checkpoints") -> list[dict[str, Any]]:
    """List checkpoints sorted by latest modification."""
    path = Path(ckpt_dir)
    if not path.exists():
        return []

    checkpoints: list[dict[str, Any]] = []
    for f in path.rglob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        checkpoints.append(
            {
                "file": f.name,
                "path": str(f),
                "size_mb": round(size_mb, 1),
                "modified": mtime.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)


def check_data_pipeline(data_dir: str | Path = "data") -> dict[str, Any]:
    """Check governed data artifacts and binary token shards."""
    path = Path(data_dir)
    stats: dict[str, Any] = {
        "exists": path.exists(),
        "total_shards": 0,
        "shards_size_mb": 0.0,
    }

    if not path.exists():
        return stats

    bin_shards = list(path.rglob("*.bin"))
    stats["total_shards"] = len(bin_shards)
    total_bytes = sum(f.stat().st_size for f in bin_shards)
    stats["shards_size_mb"] = round(total_bytes / (1024 * 1024), 2)

    governed_manifests = list(path.rglob("*manifest*.json"))
    stats["manifest_count"] = len(governed_manifests)
    return stats


def check_eval_benchmarks(eval_dir: str | Path = "eval_out") -> dict[str, Any]:
    """Inspect latest BharatBench evaluation results."""
    path = Path(eval_dir)
    report: dict[str, Any] = {"exists": path.exists(), "runs": []}
    if not path.exists():
        return report

    json_reports = sorted(path.glob("*.json"), reverse=True)
    for f in json_reports[:5]:
        try:
            with open(f, encoding="utf-8") as rf:
                data = json.load(rf)
            report["runs"].append(
                {
                    "file": f.name,
                    "model_name": data.get("model_name", "Unknown"),
                    "aggregate_score": data.get("aggregate_score", None),
                    "total_examples": data.get("total_examples", None),
                }
            )
        except Exception:
            pass
    return report


def get_full_system_status(
    ckpt_dir: str | Path = "checkpoints",
    data_dir: str | Path = "data",
    eval_dir: str | Path = "eval_out",
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "telemetry": get_hardware_telemetry(),
        "checkpoints": check_checkpoints(ckpt_dir),
        "data_pipeline": check_data_pipeline(data_dir),
        "evaluation": check_eval_benchmarks(eval_dir),
    }


def print_dashboard(status: dict[str, Any]) -> None:
    """Pretty-print terminal monitoring dashboard."""
    print("=" * 64)
    print("  🇮🇳 IndicLLM-Bharat — Infrastructure & Training Telemetry")
    print(f"  Timestamp: {status['timestamp']}")
    print("=" * 64)

    # Telemetry
    telem = status["telemetry"]
    print("\n  🖥️  Compute & Hardware Accelerator:")
    print(f"     Device        : {telem['device'].upper()}")
    print(f"     CPU Cores     : {telem['cpu_count']}")
    if telem.get("cuda_device_name"):
        print(f"     GPU Model     : {telem['cuda_device_name']}")
        print(f"     VRAM Allocated: {telem.get('cuda_memory_allocated_mb', 0)} MB")
    elif telem.get("mps_backend"):
        print(f"     MPS Backend   : {telem['mps_backend']}")

    # Checkpoints
    ckpts = status["checkpoints"]
    print(f"\n  💾 Checkpoints Found ({len(ckpts)}):")
    if ckpts:
        for c in ckpts[:3]:
            print(f"     • {c['file']:24s} | {c['size_mb']:>7.1f} MB | {c['modified']}")
    else:
        print("     • (No .pt checkpoints in target directory)")

    # Data
    data = status["data_pipeline"]
    print("\n  📦 Data Pipeline & Governance:")
    print(
        f"     Token Shards  : {data.get('total_shards', 0)} ({data.get('shards_size_mb', 0)} MB)"
    )
    print(f"     Manifests     : {data.get('manifest_count', 0)}")

    # Evaluation
    ev = status["evaluation"]
    print(f"\n  📈 Evaluation Benchmarks ({len(ev.get('runs', []))} runs):")
    if ev.get("runs"):
        for r in ev["runs"]:
            score_str = (
                f"{r['aggregate_score']:.3f}" if r.get("aggregate_score") is not None else "N/A"
            )
            print(f"     • {r['file']:24s} | Score: {score_str}")
    else:
        print("     • (No evaluation reports found)")

    print("\n" + "=" * 64 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndicLLM-Bharat Infrastructure & Telemetry Monitor"
    )
    parser.add_argument(
        "--checkpoints-dir", default="checkpoints", help="Path to checkpoints directory"
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--eval-dir", default="eval_out", help="Path to eval output directory")
    parser.add_argument("--json", action="store_true", help="Output status as JSON to stdout")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh dashboard")
    parser.add_argument(
        "--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        while True:
            status = get_full_system_status(
                ckpt_dir=args.checkpoints_dir,
                data_dir=args.data_dir,
                eval_dir=args.eval_dir,
            )

            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print_dashboard(status)

            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")
        return 0
    except Exception as e:
        print(f"error monitoring system: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
