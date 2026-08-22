# Performance Baseline Benchmark Report

**Benchmark Date**: `2026-08-22T09:03:58Z`  
**Hardware Profile**: `Apple Silicon MPS (Apple Silicon GPU (MPS))`  
**Python / PyTorch**: `3.14.5 / 2.12.1`

---

## 1. Inference Latency & Throughput Metrics

| Parameter Tier | Parameter Count | TTFT (ms) [Min / Avg / Max] | Throughput (Tokens/sec) [Avg / Max] | Total Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|
| `TINY` | 3,290,496 | 4.94 / **8.48** / 21.39 | **172.71** / 221.63 | 229.8 |

---

## 2. Knowledge Retrieval Latency Metrics

| Retrieval Subsystem | Average Latency (ms) | Min Latency (ms) | Max Latency (ms) |
|:---|:---:|:---:|:---:|
| **Dense Vector Search** | **0.43 ms** | 0.14 ms | 2.22 ms |
| **End-to-End Grounded RAG** | **135.29 ms** | - | - |

---

## 3. Latency Optimization Targets for Universal Hybrid System

| Milestone | Baseline TTFT | Optimized Target TTFT | Baseline TPS | Optimized Target TPS | Target Cache Hit Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Local Inference** | 8.48 ms | **< 35 ms** | 172.71 tok/s | **> 120 tok/s** | **< 5 ms (Exact/Semantic)** |
| **Hybrid RAG** | 0.43 ms | **< 15 ms** | - | - | **< 2 ms** |
| **AWS Bedrock Hybrid** | N/A | **< 200 ms** | - | **> 85 tok/s** | **< 20 ms (Prompt Cache)** |
