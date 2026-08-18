# 🇮🇳 Bharat AI (IndicLLM-Bharat)

**Sovereign Large Language Models for Indian Languages — Modern Decoder Architecture, Governed Indic Data Pipelines, Alignment, OpenAI-Compatible APIs, GGUF Quantization & Production Serving.**

Bharat AI delivers production-grade, open-access multilingual language models engineered for Indic languages (13+ scheduled scripts + English) with modern architecture features: **Rotary Position Embeddings (RoPE)**, **Root Mean Square Normalization (RMSNorm)**, **SwiGLU Gated Feed-Forward Networks**, and **Grouped-Query Attention (GQA)**.

> **Status**: **Milestones 1 through 7 complete** (Architecture, Data Governance Engine, BharatBench Evaluation, Serving & Quantization, Bharat-350M Pretraining, Post-Training Alignment, End-to-End Pipeline Orchestrator, Model Card, OpenAI Inference Server, Native GGUF/Ollama Exporter, Telemetry Monitor, and Production Release Tooling). **Milestone 6.1** remains gated on controlled production tokenization datasets.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![CI](https://github.com/GenixBit/IndicLLM-Bharat-V1/actions/workflows/ci.yml/badge.svg)](https://github.com/GenixBit/IndicLLM-Bharat-V1/actions/workflows/ci.yml)

---

## Quick Start

```bash
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git
cd IndicLLM-Bharat-V1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Environment & Fast Architecture Sanity Check (Native Bharat model on CPU/MPS/CUDA)
bash scripts/run_local_sanity.sh
sanity-check --model bharat --device auto

# 2. Run comprehensive test suite (2,500+ tests)
pytest tests/

# 3. Calculate model parameters and memory footprints
calculate-params --all --weight-dtype bf16
```

---

## Model Architecture & Sizing Tiers

Validated configurations are available in `configs/models/`:

| Tier | Parameters | Hidden Size | Intermediate | Layers | Heads (Q / KV) | GQA Ratio | Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bharat-350M** | 347,393,024 | 1024 | 2816 | 16 | 16 / 4 | 4:1 | 2048 |
| **Bharat-1B** | 999,368,704 | 2048 | 5632 | 24 | 32 / 8 | 4:1 | 4096 |
| **Bharat-3B** | 3,242,168,320 | 3200 | 8800 | 28 | 32 / 8 | 4:1 | 4096 |
| **Bharat-7B** | 6,864,187,392 | 4096 | 11008 | 32 | 32 / 8 | 4:1 | 4096 |

See [docs/MODEL_CONFIGURATIONS.md](docs/MODEL_CONFIGURATIONS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full architecture details.

---

## Data Governance & Preparation Engine

A cryptographic, policy-governed data-source registry lives at `data_registry/` supporting:
- **Verified Open Sources**: `indiccorp_v2` (CC-BY-4.0), `sangraha` (CC0-1.0 / MIT), `samanantar` (CC-BY-4.0), `wikipedia_indic` (CC-BY-SA-4.0).
- **Linguistic Pre-filters**: Unicode category preservation for Indic matras and Dandasa, alpha ratio filtering ($\ge 0.65$), PII scrubbing, exact and MinHash/LSH deduplication.
- **Deterministic Sharding & Manifests**: SHA-256 integrity verification, approval validation, and release packaging.

```bash
# Validate data source registry and license policies
validate-data-registry

# Ingest and prepare governed Indic data shards
prepare-indic-data \
    --source-id sangraha \
    --input-dir data/raw/sangraha \
    --output-dir data/governed/sangraha \
    --target-shard-size-mb 64

# Plan multi-source data mixtures for pretraining
plan-data-mixture \
    --recipe configs/data/mixture_pretrain_indic_1b.yaml \
    --manifests-dir data/governed/ \
    --target-tokens 1000000000

# Compute dataset statistics & build sealed release package
compute-data-stats --input data/governed/ --json
build-dataset-release --manifest data/governed/manifest.json --approval data/governed/approval.json --output-dir dist/data_release
```

---

## End-to-End Training & Alignment Pipeline

IndicLLM-Bharat provides a unified orchestrator executing the entire lifecycle: **Pretraining $\rightarrow$ SFT $\rightarrow$ DPO $\rightarrow$ BharatBench Evaluation**:

```bash
# Execute full end-to-end pipeline (dry-run mode)
run-pipeline --config configs/pipeline/bharat-350m-e2e.yaml --dry-run

# Run pretraining directly on Indic token shards
pretrain-bharat \
    --model-config configs/models/bharat-350m.yaml \
    --data-path data/indic_shards/train_0000.bin \
    --output-dir checkpoints/bharat-350m \
    --max-iters 10000 \
    --device cuda \
    --dtype bfloat16
```

---

## Local Generation & Interactive Chat REPL

```bash
# Single prompt generation
bharat-generate --checkpoint checkpoints/bharat-350m/final.pt --prompt "नमस्ते भारत" --max-tokens 64

# Interactive streaming chat REPL
bharat-generate --checkpoint checkpoints/bharat-350m/final.pt --interactive
```

---

## OpenAI-Compatible FastAPI Inference Server

Run a high-performance HTTP/REST serving endpoint supporting OpenAI-compatible chat completions (`POST /v1/chat/completions`) with real-time Server-Sent Events (SSE) token streaming:

```bash
# Launch server on port 8000
bharat-api --checkpoint checkpoints/bharat-350m/final.pt --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bharat-350m",
    "messages": [{"role": "user", "content": "भारत की राजधानी क्या है?"}],
    "stream": true
  }'
```

---

## GGUF & Ollama Export

Export checkpoints to native GGUF binaries (`F32` or quantized `Q8_0`) and generate ready-to-use Ollama `Modelfile` configs:

```bash
# Export GGUF and Modelfile for Ollama
export-ollama --checkpoint checkpoints/bharat-350m/final.pt --name bharat-350m --quant q8_0

# Run in Ollama
ollama create bharat-350m -f dist/ollama/bharat-350m/Modelfile
ollama run bharat-350m "नमस्ते! आप कैसे हैं?"
```

---

## Inference Latency & Throughput Profiler

Benchmark Time-To-First-Token (TTFT), inter-token latency (ITL), generation throughput (tok/s), and memory footprint across batch sizes and context lengths:

```bash
# Benchmark model architecture across standard sweeps
bharat-profile --model-config configs/models/bharat-350m.yaml --batch-sizes 1,2,4 --prompt-lengths 64,256 --gen-lengths 32,64

# Benchmark a trained checkpoint on GPU/MPS and emit JSON report
bharat-profile --checkpoint checkpoints/bharat-350m/final.pt --device auto --dtype bf16 --json --output profile_report.json
```

---

## Interactive Streaming Web Playground

Launch an interactive single-page web UI with real-time token streaming, 22 Indian language prompt starters, and live hyperparameter controls:

```bash
# Launch interactive playground on http://localhost:7860
bharat-playground --checkpoint checkpoints/bharat-350m/final.pt --port 7860

# Launch with synthetic tiny model for testing
bharat-playground --model-size tiny --port 7860
```

---

## Multi-GPU & Distributed Training Recipes

Production-ready Accelerate and DeepSpeed recipes are provided under `configs/distributed/`:

```bash
# Multi-GPU Distributed Data Parallel (DDP)
accelerate launch --config_file configs/distributed/accelerate_ddp.yaml scripts/pretrain_bharat.py ...

# Fully Sharded Data Parallel (FSDP with BharatDecoderLayer wrapping & BF16)
accelerate launch --config_file configs/distributed/accelerate_fsdp.yaml scripts/pretrain_bharat.py ...

# DeepSpeed ZeRO-2 / ZeRO-3 with activation checkpointing
deepspeed --config_file configs/distributed/deepspeed_zero3.yaml scripts/pretrain_bharat.py ...
```

---

## System Telemetry & Training Monitor

Inspect real-time GPU VRAM, Apple Silicon MPS unified memory, system CPU/RAM, active checkpoint steps, and ingestion pipeline progress:

```bash
# Terminal dashboard
bharat-monitor --checkpoints-dir checkpoints/ --data-dir data/governed/

# Machine-readable JSON output
bharat-monitor --json
```

---

## BharatBench Multi-Task Evaluation

Native evaluation suite measuring performance across 5 benchmark categories: **Language Understanding**, **Reasoning**, **Coding**, **Factual Knowledge**, and **Safety**:

```bash
# Evaluate checkpoint predictions on BharatBench
run-bharatbench \
    --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
    --predictions predictions.jsonl \
    --output-dir eval_out \
    --json

# Aggregate leaderboards across evaluation reports
build-leaderboard --reports-dir eval_out/
```

---

## Production Release Packaging & HF Hub Publisher

```bash
# Package production release bundle (Safetensors + GGUF + SHA-256 Manifest + Config)
build-release-bundle \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --model-config configs/models/bharat-350m.yaml \
    --tokenizer data/indic/tokenizer.json \
    --output-dir dist/bharat-350m-v1.0.0 \
    --model-name Bharat-350M \
    --include-gguf \
    --gguf-type Q8_0

# Publish release bundle directly to Hugging Face Hub
push-to-hub --bundle-dir dist/bharat-350m-v1.0.0 --repo GenixBit/IndicLLM-Bharat-350M
```

---

## Repository Structure

```
├── bharat/               # Core Bharat package
│   ├── data/             # Quality filters, deduplication, PII, manifests, mixture planner
│   ├── eval/             # BharatBench schema, metrics, runner, local inference adapter, catalog
│   ├── models/           # BharatForCausalLM (GQA, RoPE, RMSNorm, SwiGLU) and configs
│   ├── posttraining/     # SFT (assistant loss masking), DPO (preference loss)
│   ├── serving/          # Safetensors/GGUF exporters, streaming, auth, rate limiting, metrics
│   ├── tokenizer/        # Unified BharatTokenizer interface and BPE training harness
│   └── training/         # Pretrain engine, optimizer partitioner, checkpointing, E2E pipeline
├── configs/              # Model, data mixture, and pipeline recipes
├── data_registry/        # Versioned governed data sources and license policy
├── docs/                 # Architectural specifications, roadmap, Model Card, release guides
├── eval_fixtures/        # Evaluation test fixtures and leaderboard benchmarks
├── inference/            # Interactive generation REPL, FastAPI OpenAI server, Ollama export
├── scripts/              # Command-line tools (pretrain, prepare-data, plan-mixture, pipeline, release)
└── tests/                # 2,500+ unit and integration tests (100% pass rate)
```

---

## License & Attribution

Licensed under the [Apache License, Version 2.0](LICENSE).
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for citation and attribution details.
