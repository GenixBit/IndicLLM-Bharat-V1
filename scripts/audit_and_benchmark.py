# ruff: noqa: E402
"""Full System Architecture Audit & Performance Baseline Benchmarking Suite."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat.rag.engine import SovereignRAGPipeline
from bharat.serving.openai_server import BharatInferenceEngine


def get_environment_info() -> dict[str, Any]:
    """Inspect compute hardware and execution environment safely."""
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["accelerator"] = "CUDA"
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["accelerator"] = "Apple Silicon MPS"
        info["device_name"] = "Apple Silicon GPU (MPS)"
        info["mps_built"] = torch.backends.mps.is_built()
    else:
        info["accelerator"] = "CPU"
        info["device_name"] = platform.processor() or "CPU"

    aws_cli_installed = False
    aws_configured = False
    try:
        res = subprocess.run(["aws", "--version"], capture_output=True, text=True, timeout=3)
        aws_cli_installed = res.returncode == 0
        info["aws_cli_version"] = res.stdout.strip() or res.stderr.strip()
    except Exception:
        info["aws_cli_version"] = "Not Installed"

    try:
        res = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            aws_data = json.loads(res.stdout)
            aws_configured = True
            info["aws_identity_arn"] = aws_data.get("Arn", "Authenticated")
            info["aws_account_id"] = "******" + aws_data.get("Account", "0000")[-4:]
    except Exception:
        pass

    info["aws_status"] = (
        "Configured"
        if aws_configured
        else ("CLI Available" if aws_cli_installed else "Not Configured")
    )
    return info


def benchmark_inference(
    tier: str = "tiny", device: str = "cpu", num_trials: int = 5
) -> dict[str, Any]:
    """Measure TTFT, TPS, prompt processing, and total generation latency."""
    engine = BharatInferenceEngine(tier=tier, device=device)
    tokenizer = engine.tokenizer

    prompts = [
        "What is the significance of the Constitution of India in promoting national unity and scientific temper?",
        "Explain the fundamental difference between classical and quantum computing algorithms.",
        "భారతీయ అంతరిక్ష పరిశోధన సంస్థ (ISRO) విజయాల గురించి వివరించండి.",
    ]

    ttft_list: list[float] = []
    tps_list: list[float] = []
    total_latency_list: list[float] = []

    # Warmup
    _ = engine.generate("Warmup prompt", max_new_tokens=5)

    for p in prompts:
        for _ in range(num_trials):
            start_prompt = time.perf_counter()
            first_token_time = None
            generated_chunks: list[str] = []

            for chunk in engine.generate_stream(p, max_new_tokens=32, temperature=0.7):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_chunks.append(chunk)

            end_gen = time.perf_counter()

            ttft = (first_token_time - start_prompt) * 1000.0 if first_token_time else 0.0
            total_time = end_gen - start_prompt
            gen_time = end_gen - (first_token_time or start_prompt)
            gen_tokens = len(tokenizer.encode("".join(generated_chunks)))
            tps = gen_tokens / max(1e-5, gen_time)

            ttft_list.append(ttft)
            tps_list.append(tps)
            total_latency_list.append(total_time * 1000.0)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_tps = sum(tps_list) / len(tps_list)
    avg_total_lat = sum(total_latency_list) / len(total_latency_list)

    return {
        "tier": tier,
        "device": str(engine.device),
        "param_count": sum(p.numel() for p in engine.model.parameters()),
        "avg_ttft_ms": round(avg_ttft, 2),
        "min_ttft_ms": round(min(ttft_list), 2),
        "max_ttft_ms": round(max(ttft_list), 2),
        "avg_tps": round(avg_tps, 2),
        "max_tps": round(max(tps_list), 2),
        "avg_total_latency_ms": round(avg_total_lat, 2),
        "sample_trials": len(ttft_list),
    }


def benchmark_rag_retrieval(tier: str = "tiny", device: str = "cpu") -> dict[str, Any]:
    """Measure embedding generation and vector search latency."""
    pipeline = SovereignRAGPipeline(tier=tier, device=device)
    queries = [
        "What are the Fundamental Rights in the Constitution of India?",
        "Tell me about ISRO Chandrayaan-3 lunar landing",
        "Explain National Quantum Mission",
    ]

    retrieval_latencies: list[float] = []
    e2e_rag_latencies: list[float] = []

    for q in queries:
        for _ in range(5):
            t0 = time.perf_counter()
            _ = pipeline.index.search(q, top_k=2)
            t1 = time.perf_counter()
            _ = pipeline.query(q, top_k=2, max_new_tokens=16)
            t2 = time.perf_counter()

            retrieval_latencies.append((t1 - t0) * 1000.0)
            e2e_rag_latencies.append((t2 - t0) * 1000.0)

    return {
        "avg_vector_search_ms": round(sum(retrieval_latencies) / len(retrieval_latencies), 2),
        "min_vector_search_ms": round(min(retrieval_latencies), 2),
        "max_vector_search_ms": round(max(retrieval_latencies), 2),
        "avg_e2e_rag_latency_ms": round(sum(e2e_rag_latencies) / len(e2e_rag_latencies), 2),
    }


def run_full_audit() -> dict[str, Any]:
    """Execute complete audit and generate report structure."""
    print("=" * 65)
    print("🔍 IndicLLM-Bharat Full Architecture Audit & Benchmarking Suite")
    print("=" * 65)

    env_info = get_environment_info()
    print(f"  • Operating System: {env_info['os']} ({env_info['architecture']})")
    print(f"  • Accelerator:      {env_info['accelerator']} ({env_info['device_name']})")
    print(f"  • AWS Status:       {env_info['aws_status']}")

    print("\n⏳ Benchmarking Base Inference Latency...")
    infer_tiny = benchmark_inference(tier="tiny", device="cpu", num_trials=3)
    print(
        f"  • Tiny Tier TTFT:   {infer_tiny['avg_ttft_ms']} ms | TPS: {infer_tiny['avg_tps']} tok/s"
    )

    print("\n⏳ Benchmarking Vector Retrieval Latency...")
    rag_bench = benchmark_rag_retrieval(tier="tiny", device="cpu")
    print(
        f"  • Vector Search:    {rag_bench['avg_vector_search_ms']} ms | E2E RAG: {rag_bench['avg_e2e_rag_latency_ms']} ms"
    )

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env_info,
        "inference_benchmarks": {
            "tiny": infer_tiny,
        },
        "retrieval_benchmarks": rag_bench,
        "identified_bottlenecks": [
            {
                "subsystem": "Inference Engine",
                "bottleneck": "Iterative autoregressive tensor concatenation in Python loop creates per-token GPU/CPU memory allocations.",
                "remedy": "Dynamic continuous KV Caching with pre-allocated buffer tensors and static memory pool.",
            },
            {
                "subsystem": "Model Routing",
                "bottleneck": "System lacks an intelligent router to delegate complex queries to cloud (AWS Bedrock) or tools.",
                "remedy": "Implement Multi-Tier AI Router with intent classification, complexity scoring, and cost/latency estimation.",
            },
            {
                "subsystem": "Knowledge Retrieval",
                "bottleneck": "Pure dense vector search misses exact keyword/lexical terms and lacks multi-hop entity reasoning.",
                "remedy": "Build Hybrid Search (Dense + BM25 Lexical + Reciprocal Rank Fusion + Knowledge Graph).",
            },
            {
                "subsystem": "Live Data Freshness",
                "bottleneck": "Static model context cannot answer real-time/latest 2026 information without web intelligence.",
                "remedy": "Build Live Web Intelligence subsystem with robots.txt compliance, multi-source extraction, and fact verification.",
            },
            {
                "subsystem": "Caching & Cost",
                "bottleneck": "Repeated queries trigger full autoregressive inference without caching.",
                "remedy": "Deploy multi-tier caching (Exact Match, Semantic Cache, Prompt Context Cache, Tool Cache).",
            },
        ],
    }

    return audit_summary


def export_audit_documents(audit: dict[str, Any]) -> tuple[Path, Path]:
    """Export ARCHITECTURE_AUDIT.md and PERFORMANCE_BASELINE.md."""
    doc_dir = ROOT / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)

    audit_md = ROOT / "ARCHITECTURE_AUDIT.md"
    perf_md = ROOT / "PERFORMANCE_BASELINE.md"

    env = audit["environment"]
    inf = audit["inference_benchmarks"]["tiny"]
    rag = audit["retrieval_benchmarks"]

    audit_content = f"""# System Architecture Audit Report: IndicLLM-Bharat

**Audit Timestamp**: `{audit['timestamp']}`
**Operating Environment**: `{env['os']} {env['os_release']} ({env['architecture']})`
**Compute Accelerator**: `{env['accelerator']}` ({env['device_name']})
**AWS Integration Status**: `{env['aws_status']}`

---

## 1. Executive Summary & Existing Subsystems

IndicLLM-Bharat is currently structured as an end-to-end sovereign foundation LLM platform containing:
- **Foundation Model Architecture**: `BharatForCausalLM` with Grouped-Query Attention (GQA), SwiGLU, RMSNorm, and YaRN RoPE (32k context).
- **Tokenization**: `BharatTokenizer` supporting 22 Scheduled Indian Languages + English (50k - 64k vocabulary).
- **Inference & Serving**: Standard HTTP/SSE streaming server (`bharat/serving/openai_server.py`) and Web Playground UI.
- **RAG & Tools**: Basic in-memory dense vector index (`bharat/rag/engine.py`) and safe Python/unit converter tool executor.
- **Safety & Constitutional AI**: Self-critique synthesis (`bharat/synthetic/constitutional.py`) and guardrail auditing.

---

## 2. Comprehensive Bottleneck Analysis

| Subsystem | Identified Bottleneck | Architectural Impact | Target Production Remedy |
|:---|:---|:---|:---|
| **Inference Engine** | Per-step tensor concatenation in generation loop | High CPU/VRAM reallocation overhead | Pre-allocated continuous KV Cache & Prompt Caching |
| **Model Routing** | Monolithic local model fallback only | High latency / failure on complex queries | Hybrid AI Router (Local $\\leftrightarrow$ AWS Bedrock $\\leftrightarrow$ Web) |
| **Knowledge Retrieval** | Pure dense vector search | Misses exact keywords (acronyms, names, codes) | Hybrid Retrieval (Dense Vector + BM25 + Knowledge Graph) |
| **Data Freshness** | Static knowledge baseline | Outdated responses on real-time queries | Live Web Retrieval & Continuous Ingestion Engine |
| **Tool Ecosystem** | Limited hardcoded tool functions | Lack of standard discovery protocol | MCP-Compatible Tool Registry & Multi-Step ReAct Loop |
| **Fact Verification** | No cross-source validation | Potential hallucination on conflicting data | Multi-Source Fact Checking & Grounded Citation Engine |
| **Caching Layer** | Zero query or semantic cache | Repeated compute & token waste | Exact, Semantic, Retrieval, and Prompt Caching |

---

## 3. Recommended Hybrid Architecture

```mermaid
graph TD
    User[User Query in 22 Indic Languages or English] --> Router[Intelligent AI Model Router]
    Router -->|Simple / Private| Local[Local Bharat LLM with Dynamic KV Cache]
    Router -->|Complex / Long Context| Cloud[AWS Bedrock High-Performance Models]
    Router -->|Latest / Real-Time News| Web[Live Web Intelligence & Verified Extraction]
    Router -->|Calculation / API| Tools[MCP Safe Tool Execution Engine]

    Local & Cloud & Web & Tools --> HybridRAG[Hybrid Search: Dense Vector + BM25 + Knowledge Graph]
    HybridRAG --> Verifier[Fact Verification & Citation Engine]
    Verifier --> Stream[Real-Time Token Streaming Output]
```
"""

    perf_content = f"""# Performance Baseline Benchmark Report

**Benchmark Date**: `{audit['timestamp']}`
**Hardware Profile**: `{env['accelerator']} ({env['device_name']})`
**Python / PyTorch**: `{env['python_version']} / {env['pytorch_version']}`

---

## 1. Inference Latency & Throughput Metrics

| Parameter Tier | Parameter Count | TTFT (ms) [Min / Avg / Max] | Throughput (Tokens/sec) [Avg / Max] | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|
| `TINY` | {inf['param_count']:,} | {inf['min_ttft_ms']} / **{inf['avg_ttft_ms']}** / {inf['max_ttft_ms']} | **{inf['avg_tps']}** / {inf['max_tps']} | {inf['avg_total_latency_ms']} |

---

## 2. Knowledge Retrieval Latency Metrics

| Retrieval Subsystem | Average Latency (ms) | Min Latency (ms) | Max Latency (ms) |
|:---|:---:|:---:|:---:|
| **Dense Vector Search** | **{rag['avg_vector_search_ms']} ms** | {rag['min_vector_search_ms']} ms | {rag['max_vector_search_ms']} ms |
| **End-to-End Grounded RAG** | **{rag['avg_e2e_rag_latency_ms']} ms** | - | - |

---

## 3. Latency Optimization Targets for Universal Hybrid System

| Milestone | Baseline TTFT | Optimized Target TTFT | Baseline TPS | Optimized Target TPS | Target Cache Hit Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Local Inference** | {inf['avg_ttft_ms']} ms | **< 35 ms** | {inf['avg_tps']} tok/s | **> 120 tok/s** | **< 5 ms (Exact/Semantic)** |
| **Hybrid RAG** | {rag['avg_vector_search_ms']} ms | **< 15 ms** | - | - | **< 2 ms** |
| **AWS Bedrock Hybrid** | N/A | **< 200 ms** | - | **> 85 tok/s** | **< 20 ms (Prompt Cache)** |
"""

    with open(audit_md, "w", encoding="utf-8") as f:
        f.write(audit_content)

    with open(perf_md, "w", encoding="utf-8") as f:
        f.write(perf_content)

    return audit_md, perf_md


def main() -> int:
    audit = run_full_audit()
    audit_md, perf_md = export_audit_documents(audit)

    print("\n" + "=" * 65)
    print("✅ Full Architecture Audit & Performance Baseline Complete!")
    print(f"  • Architecture Audit:     {audit_md.resolve()}")
    print(f"  • Performance Baseline:   {perf_md.resolve()}")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
