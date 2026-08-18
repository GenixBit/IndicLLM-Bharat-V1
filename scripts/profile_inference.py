#!/usr/bin/env python3
"""IndicLLM-Bharat — Inference Latency, Throughput & Memory Profiler CLI.

Profiles prefill latency (Time To First Token / TTFT), inter-token latency (ITL),
generation throughput (tokens/sec), and memory footprint across batch sizes and sequence lengths.

Usage:
  # Profile a model config directly
  python scripts/profile_inference.py --model-config configs/models/bharat-350m.yaml

  # Profile specific sweep matrices
  python scripts/profile_inference.py --model-size 350m --batch-sizes 1,2 --prompt-lengths 64,256 --gen-lengths 32

  # Output JSON report
  python scripts/profile_inference.py --model-size 350m --json --output profile_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from bharat.models.bharat_model import BharatForCausalLM  # noqa: E402
from bharat.models.config import BharatModelConfig  # noqa: E402
from bharat.models.sizing import calculate_kv_cache_memory  # noqa: E402


@dataclass
class ProfileResult:
    """Individual benchmark run result for a given configuration."""

    batch_size: int
    prompt_length: int
    gen_length: int
    ttft_ms: float
    avg_itl_ms: float
    total_time_s: float
    decode_time_s: float
    tokens_generated: int
    gen_throughput_tok_per_s: float
    total_throughput_tok_per_s: float
    kv_cache_mb: float
    peak_memory_mb: float


@dataclass
class ProfilingReport:
    """Complete profiling report across all evaluated configurations."""

    model_name: str
    num_parameters: int
    device: str
    dtype: str
    timestamp: str
    results: list[ProfileResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_parameters": self.num_parameters,
            "device": self.device,
            "dtype": self.dtype,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak memory allocation in MB for the current device."""
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        with contextlib.suppress(Exception):
            return float(torch.mps.current_allocated_memory() / (1024 * 1024))
        return 0.0
    # Fallback for CPU
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes, on Linux in KB
        if sys.platform == "darwin":
            return float(usage.ru_maxrss / (1024 * 1024))
        return float(usage.ru_maxrss / 1024)
    except Exception:
        return 0.0


def sync_device(device: torch.device) -> None:
    """Synchronize device streams for accurate timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        with contextlib.suppress(Exception):
            torch.mps.synchronize()


@torch.no_grad()
def benchmark_single_config(
    model: BharatForCausalLM,
    config: BharatModelConfig,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    device: torch.device,
    warmup: int = 2,
    runs: int = 3,
) -> ProfileResult:
    """Benchmark a single (batch_size, prompt_len, gen_len) setting."""
    model.eval()

    # Generate synthetic input prompt tokens
    vocab_size = config.vocab_size
    input_ids = torch.randint(
        0, vocab_size, (batch_size, prompt_len), dtype=torch.long, device=device
    )

    # Warmup runs
    for _ in range(max(1, warmup)):
        # Prefill
        out = model(input_ids)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_ids = torch.cat([input_ids, next_token], dim=1)
        # Short generate
        for _ in range(min(4, gen_len)):
            out = model(cur_ids)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
        sync_device(device)

    # Measured runs
    ttft_samples: list[float] = []
    total_time_samples: list[float] = []
    decode_time_samples: list[float] = []
    itl_samples: list[float] = []

    for _ in range(max(1, runs)):
        sync_device(device)
        t_start = time.perf_counter()

        # Step 1: Prefill (TTFT)
        out = model(input_ids)
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        sync_device(device)
        t_ttft = time.perf_counter()

        ttft_ms = (t_ttft - t_start) * 1000.0
        ttft_samples.append(ttft_ms)

        # Step 2: Decode loop
        curr_ids = torch.cat([input_ids, next_tok], dim=1)
        step_times: list[float] = []
        t_prev = t_ttft

        for _ in range(gen_len - 1):
            out = model(curr_ids)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            sync_device(device)
            t_curr = time.perf_counter()
            step_times.append((t_curr - t_prev) * 1000.0)
            t_prev = t_curr
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)

        t_end = time.perf_counter()
        total_time = t_end - t_start
        decode_time = t_end - t_ttft

        total_time_samples.append(total_time)
        decode_time_samples.append(decode_time)
        if step_times:
            itl_samples.append(sum(step_times) / len(step_times))
        else:
            itl_samples.append(ttft_ms)

    avg_ttft_ms = sum(ttft_samples) / len(ttft_samples)
    avg_total_time = sum(total_time_samples) / len(total_time_samples)
    avg_decode_time = sum(decode_time_samples) / len(decode_time_samples)
    avg_itl_ms = sum(itl_samples) / len(itl_samples)

    tokens_generated = batch_size * gen_len
    gen_throughput = (
        tokens_generated / max(avg_decode_time, 1e-6)
        if gen_len > 1
        else (tokens_generated / max(avg_total_time, 1e-6))
    )
    total_tokens = batch_size * (prompt_len + gen_len)
    total_throughput = total_tokens / max(avg_total_time, 1e-6)

    # KV Cache memory calculation
    try:
        kv_cache_bytes = calculate_kv_cache_memory(
            config=config,
            batch_size=batch_size,
            seq_len=prompt_len + gen_len,
            dtype="bf16",
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
    except Exception:
        kv_cache_mb = 0.0

    peak_memory = get_peak_memory_mb(device)

    return ProfileResult(
        batch_size=batch_size,
        prompt_length=prompt_len,
        gen_length=gen_len,
        ttft_ms=round(avg_ttft_ms, 2),
        avg_itl_ms=round(avg_itl_ms, 2),
        total_time_s=round(avg_total_time, 4),
        decode_time_s=round(avg_decode_time, 4),
        tokens_generated=tokens_generated,
        gen_throughput_tok_per_s=round(gen_throughput, 2),
        total_throughput_tok_per_s=round(total_throughput, 2),
        kv_cache_mb=round(kv_cache_mb, 2),
        peak_memory_mb=round(peak_memory, 2),
    )


def profile_model(
    model: BharatForCausalLM,
    config: BharatModelConfig,
    batch_sizes: list[int],
    prompt_lengths: list[int],
    gen_lengths: list[int],
    device_name: str = "auto",
    dtype_name: str = "bf16",
    warmup: int = 2,
    runs: int = 3,
    model_name: str = "BharatForCausalLM",
) -> ProfilingReport:
    """Run a full profiling suite across the parameter sweep matrix."""
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    torch_dtype = torch.float32
    if dtype_name in ("bf16", "bfloat16") and device.type != "cpu":
        torch_dtype = torch.bfloat16
    elif dtype_name in ("fp16", "float16") and device.type != "cpu":
        torch_dtype = torch.float16

    model.to(device=device, dtype=torch_dtype)

    num_params = sum(p.numel() for p in model.parameters())
    results: list[ProfileResult] = []

    for bs in batch_sizes:
        for p_len in prompt_lengths:
            for g_len in gen_lengths:
                res = benchmark_single_config(
                    model=model,
                    config=config,
                    batch_size=bs,
                    prompt_len=p_len,
                    gen_len=g_len,
                    device=device,
                    warmup=warmup,
                    runs=runs,
                )
                results.append(res)

    return ProfilingReport(
        model_name=model_name,
        num_parameters=num_params,
        device=str(device),
        dtype=dtype_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        results=results,
    )


def format_markdown_table(report: ProfilingReport) -> str:
    """Format profiling results as a formatted markdown table."""
    lines = [
        f"### ⚡ Inference Profile Report: {report.model_name}",
        f"- **Parameters**: {report.num_parameters:,}",
        f"- **Device**: `{report.device}` | **Precision**: `{report.dtype}`",
        f"- **Generated at**: {report.timestamp}",
        "",
        "| Batch | Prompt Len | Gen Len | TTFT (ms) | ITL (ms) | Gen Throughput (tok/s) | Total Time (s) | KV Cache (MB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.batch_size} | {r.prompt_length} | {r.gen_length} | {r.ttft_ms:.1f} | "
            f"{r.avg_itl_ms:.1f} | **{r.gen_throughput_tok_per_s:.1f}** | {r.total_time_s:.3f}s | {r.kv_cache_mb:.1f} MB |"
        )
    return "\n".join(lines)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile inference latency, throughput, and memory for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML model configuration (e.g. configs/models/bharat-350m.yaml)",
    )
    group.add_argument(
        "--model-size",
        choices=["350m", "1b", "3b", "7b", "tiny"],
        default="350m",
        help="Standard Bharat model configuration tier",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained PyTorch checkpoint (.pt)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2",
        help="Comma-separated batch sizes to benchmark",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default="64,256",
        help="Comma-separated prompt sequence lengths",
    )
    parser.add_argument(
        "--gen-lengths",
        type=str,
        default="32,64",
        help="Comma-separated generation token lengths",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Target hardware device",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="Precision data type",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured benchmark iterations per setting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output to stdout",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to JSON file",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    batch_sizes = [int(b.strip()) for b in parsed.batch_sizes.split(",") if b.strip()]
    prompt_lengths = [int(p.strip()) for p in parsed.prompt_lengths.split(",") if p.strip()]
    gen_lengths = [int(g.strip()) for g in parsed.gen_lengths.split(",") if g.strip()]

    # Resolve Model Configuration
    if parsed.checkpoint:
        ckpt_p = Path(parsed.checkpoint)
        if not ckpt_p.is_file():
            print(f"Error: Checkpoint file not found: {ckpt_p}", file=sys.stderr)
            return 1
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
            cfg = BharatModelConfig.from_dict(ckpt["metadata"].model_config)
        elif "model_config" in ckpt:
            m_cfg = ckpt["model_config"]
            cfg = BharatModelConfig.from_dict(m_cfg if isinstance(m_cfg, dict) else m_cfg.__dict__)
        else:
            cfg = BharatModelConfig()
        model = BharatForCausalLM(cfg)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        model_name = f"Bharat-{ckpt_p.stem}"
    elif parsed.model_config:
        cfg_path = Path(parsed.model_config)
        if not cfg_path.is_file():
            print(f"Error: Model config file not found: {cfg_path}", file=sys.stderr)
            return 1
        cfg = BharatModelConfig.from_yaml(cfg_path)
        model = BharatForCausalLM(cfg)
        model_name = f"Bharat-{cfg_path.stem}"
    else:
        tier = parsed.model_size
        if tier == "tiny":
            cfg = BharatModelConfig(
                vocab_size=1000,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
            model_name = "Bharat-Tiny"
        else:
            yaml_path = ROOT_DIR / "configs" / "models" / f"bharat-{tier}.yaml"
            if yaml_path.is_file():
                cfg = BharatModelConfig.from_yaml(yaml_path)
            else:
                cfg = BharatModelConfig()
            model_name = f"Bharat-{tier.upper()}"
        model = BharatForCausalLM(cfg)

    report = profile_model(
        model=model,
        config=cfg,
        batch_sizes=batch_sizes,
        prompt_lengths=prompt_lengths,
        gen_lengths=gen_lengths,
        device_name=parsed.device,
        dtype_name=parsed.dtype,
        warmup=parsed.warmup,
        runs=parsed.runs,
        model_name=model_name,
    )

    if parsed.output:
        out_p = Path(parsed.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"✓ Saved profile report to {out_p}")

    if parsed.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n" + format_markdown_table(report) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
