# 🇮🇳 Bharat AI (IndicLLM-Bharat)

**Sovereign Large Language Models for Indian Languages — Modern Decoder Architecture, Governed Indic Data Pipelines, Alignment & Production Serving.**

Bharat AI delivers production-grade, open-access multilingual language models engineered for Indic languages (13+ scheduled scripts + English) with modern architecture features: **Rotary Position Embeddings (RoPE)**, **Root Mean Square Normalization (RMSNorm)**, **SwiGLU Gated Feed-Forward Networks**, and **Grouped-Query Attention (GQA)**.

> **Status**: **Milestones 1 through 7 complete** (Architecture, Data Governance Engine, BharatBench Evaluation, Serving & Quantization, Bharat-350M Pretraining, Post-Training Alignment, End-to-End Pipeline Orchestrator, Model Card, and Production Release Tooling). **Milestone 6.1** remains gated on controlled production tokenization datasets.

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

# 1. Environment & Architecture Sanity Check (Native Bharat model on CPU/MPS/CUDA)
python scripts/sanity_check.py --model bharat --max-iters 10

# 2. Run test suite
pytest tests/

# 3. Calculate model parameters and memory footprints
python scripts/calculate_params.py --all --weight-dtype bf16
```

---

## Model Architecture & Sizing Tiers

Validated configurations are available in `configs/models/`:

| Tier | Parameters | Hidden Size | Intermediate | Layers | Heads (Q / KV) | GQA Ratio | Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
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
- **Deterministic Sharding & Manifests**: SHA-256 integrity verification.

```bash
# Validate data source registry and license policies
python scripts/validate_data_registry.py

# Ingest and prepare governed Indic data shards
python scripts/prepare_indic_data.py \
    --source-id sangraha \
    --input-dir data/raw/sangraha \
    --output-dir data/governed/sangraha \
    --target-shard-size-mb 64

# Plan multi-source data mixtures for pretraining
python scripts/plan_data_mixture.py \
    --recipe configs/data/mixture_pretrain_indic_1b.yaml \
    --manifests-dir data/governed/ \
    --target-tokens 1000000000
```

---

## End-to-End Training & Alignment Pipeline

IndicLLM-Bharat provides a unified orchestrator executing the entire lifecycle: **Pretraining $\rightarrow$ SFT $\rightarrow$ DPO $\rightarrow$ BharatBench Evaluation**:

```bash
# Execute full end-to-end pipeline (dry-run mode)
python scripts/run_pipeline.py --config configs/pipeline/bharat-350m-e2e.yaml --dry-run

# Run pretraining directly on Indic token shards
python scripts/pretrain_bharat.py \
    --model-config configs/models/bharat-350m.yaml \
    --data-path data/indic_shards/train_0000.bin \
    --output-dir checkpoints/bharat-350m \
    --max-iters 10000 \
    --device cuda \
    --dtype bfloat16
```

---

## BharatBench Multi-Task Evaluation

Native evaluation suite measuring performance across 5 benchmark categories: **Language Understanding**, **Reasoning**, **Coding**, **Factual Knowledge**, and **Safety**:

```bash
# Evaluate checkpoint predictions on BharatBench
python scripts/run_bharatbench.py \
    --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
    --predictions predictions.jsonl \
    --output-dir eval_out \
    --json
```

---

## Production Serving & Quantization

Supports high-throughput local inference, streaming APIs, and deployment exports in **Safetensors** and **GGUF** (Q8_0 / F32):

```bash
# Package production release bundle with Safetensors, GGUF, config, and SHA-256 manifest
python scripts/build_release_bundle.py \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --model-config configs/models/bharat-350m.yaml \
    --tokenizer data/indic/tokenizer.json \
    --output-dir dist/bharat-350m-v1.0.0 \
    --model-name Bharat-350M \
    --include-gguf \
    --gguf-type Q8_0
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
├── scripts/              # Command-line tools (pretrain, prepare-data, plan-mixture, pipeline, release)
└── tests/                # 2,400+ unit and integration tests (100% pass rate)
```

---

## License & Attribution

Licensed under the [Apache License, Version 2.0](LICENSE).
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for citation and attribution details.
