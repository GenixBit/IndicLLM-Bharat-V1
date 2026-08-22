# System Evaluation, Benchmark Suites & Regression Framework

This document outlines the evaluation framework across factual accuracy, hybrid retrieval quality, latency, routing accuracy, and safety.

---

## 1. Automated Evaluation Matrix

| Evaluation Benchmark | Script / Tool | Focus Area | Success Criterion |
|:---|:---|:---|:---:|
| **Inference Latency & TTFT** | `scripts/audit_and_benchmark.py` | Engine throughput & memory allocation | TTFT $< 35\text{ ms}$, TPS $> 120\text{ tok/s}$ |
| **Model Routing Accuracy** | `pytest tests/routing/` | Classification of intent & freshness | $> 95\%$ correct destination match |
| **Hybrid RAG Precision** | `pytest tests/rag/` | Dense + BM25 Reciprocal Rank Fusion | High-relevance document top-1 retrieval |
| **Safety & Constitutional AI** | `scripts/evaluate_safety.py` | Sovereign guardrail & ethical adherence | Zero harmful or illegal advice |
| **Reasoning & CoT** | `scripts/evaluate_reasoning.py` | Step-by-step mathematical proofs & logic | Valid `<think>...</think><answer>` format |

---

## 2. Regression Testing Runbook

To run the complete automated test and evaluation suite before deployment:
```bash
pytest tests/
```
