# 🇮🇳 Bharat AI

**LLM for Indian languages — from scratch, validated, production-grade.**

Bharat AI is transitioning from a GPT-2 based prototype to a modern decoder architecture (RoPE, RMSNorm, SwiGLU, GQA) with unified tokenizer, proper evaluation, and production serving.

> **Status**: Milestone 1 ✅, Milestone 2.1 ✅ (components), Milestone 2.2 ✅ (full model + generation) — see [vision](docs/VISION.md), [roadmap](docs/ROADMAP.md), and [implementation plan](docs/IMPLEMENTATION_PLAN.md) for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![CI](https://github.com/GenixBit/IndicLLM-Bharat-V1/actions/workflows/ci.yml/badge.svg)](https://github.com/GenixBit/IndicLLM-Bharat-V1/actions/workflows/ci.yml)

---

## Quick Start

```bash
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git
cd IndicLLM-Bharat-V1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Sanity check — GPT-2 10M pretrain
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 500

# Run tests
pytest tests/
```

---

## Bharat Model Configurations

Four validated configurations are available in `configs/models/`:
**Bharat-350M**, **Bharat-1B**, **Bharat-3B**, and **Bharat-7B** — all using
RoPE, RMSNorm, SwiGLU, and GQA with analytical parameter counts within 1 %
of their nominal tier.

```bash
# View parameter breakdown for all configurations
python scripts/calculate_params.py --all --weight-dtype bf16
```

See [docs/MODEL_CONFIGURATIONS.md](docs/MODEL_CONFIGURATIONS.md) for full
architecture tables, parameter counts, and memory calculations.

### Truthful statements

- The 64 000 vocabulary size is an **architecture assumption**; the final Bharat tokenizer has not yet been trained.
- No RoPE interpolation, NTK scaling, YaRN or LongRoPE has been implemented.
- Model-quality impact of GQA has not been measured.
- No Bharat model has been pretrained or benchmarked.

## Data Source Registry

A versioned, governed data-source registry exists at `data_registry/`.
It enforces default-deny licensing, immutable revisions, SHA-256 integrity
pins, and deterministic ordering/digest — see
[docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

**Important:** The registry infrastructure is in place, but no dataset is
automatically legally approved. No data has been downloaded or processed.
Sharding, manifests, dataset statistics, mixture planning, and contanimation
checks are now implemented as offline utilities (Milestone 3.3). The legacy
`data/` pipelines are unchanged.

### Data Processing Pipeline

Heuristic offline filters for normalization, language identification,
quality scoring, deduplication (exact + fuzzy), PII detection, and safety
filtering live under `bharat/data/`. These are **heuristic pre-filters
only** — they are not legal, safety, or quality guarantees.

```python
from bharat.data.processing import DataProcessor

processor = DataProcessor()
decision = processor.process("your text here")
print(decision.accepted, decision.reasons)
```

```bash
# Validate the registry
python scripts/validate_data_registry.py

# Work with dataset manifests
python scripts/validate_data_manifest.py --manifest manifest.json
python scripts/plan_data_shards.py --manifest manifest.json
python scripts/compute_data_stats.py --input data.txt

# Generate deterministic local BharatBench predictions
python scripts/generate_bharatbench_predictions.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --output predictions.jsonl \
  --adapter expected \
  --json

# Evaluate predictions with BharatBench
python scripts/run_bharatbench.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --predictions predictions.jsonl \
  --output-dir eval_out \
  --json

# Generate predictions using a local model checkpoint
python scripts/generate_bharatbench_local_predictions.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --output predictions.jsonl \
  --checkpoint /path/to/checkpoint \
  --tokenizer /path/to/tokenizer \
  --max-new-tokens 256 \
  --device cpu \
  --json
```

**Important:** BharatBench evaluates local prediction files only. No model training or benchmark downloads are included yet. The tiny fixtures under `eval_fixtures/bharatbench_tiny/` are synthetic smoke tests — see [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for details.

**Milestone 4.2 note:** `scripts/generate_bharatbench_predictions.py` uses deterministic local adapters (`expected`, `echo`, and `choice-baseline`) to create prediction JSONL files for smoke testing. These adapters do not load models, do not call external APIs, do not download benchmarks, and do not perform real model generation.

**Milestone 4.3 note:** `scripts/generate_bharatbench_local_predictions.py` uses the `LocalCausalLMAdapter` to connect BharatBench predictions to a local checkpoint and tokenizer. Only local filesystem paths are accepted. Remote URLs (`http://`, `https://`, `ftp://`, `s3://`, `gs://`) are rejected. See [docs/MILESTONE_4_3_LOCAL_INFERENCE.md](docs/MILESTONE_4_3_LOCAL_INFERENCE.md) for details.

**Milestone 4.4 note:** The `BenchmarkCatalog` registers five benchmark categories — language, reasoning, coding, knowledge, and safety — each with a `safety_boundary` and supported task types. Tiny synthetic fixtures live under `eval_fixtures/benchmarks/`. See [docs/MILESTONE_4_4_BENCHMARK_CATALOG.md](docs/MILESTONE_4_4_BENCHMARK_CATALOG.md) for details.

**Milestone 4.5 note:** The `Leaderboard` aggregates BharatBench evaluation reports into ranked tables with JSON and Markdown export. The `build-leaderboard` CLI scans a directory of report JSON files. Synthetic fixtures live under `eval_fixtures/leaderboard/`. See [docs/MILESTONE_4_5_LEADERBOARD.md](docs/MILESTONE_4_5_LEADERBOARD.md) for details.

---

## Verified Results — GPT-2 10M (Training Complete)

| Metric | Iter 2000 |
|--------|-----------|
| Val Perplexity | **167.00** |
| Val Accuracy | **22.55%** |
| Gen Speed | **64 tok/s** |
| Parameters | 30.1M (6L / 384d / 6H) |
| Training data | FineWeb-Edu 51M tokens |
| Training time | ~10h on AWS c5.2xlarge |

---

## Project Structure

```
├── bharat/               # New Bharat AI package (tokenizer, training, post-training)
├── train/                # Legacy GPT-2 training (pretrain, SFT, DPO)
├── eval/                 # Evaluation benchmarks
├── inference/            # Inference and API
├── data/                 # Data pipelines (FineWeb-Edu, Indic)
├── configs/              # Model configurations
├── scripts/              # Utilities (sanity check, HF push)
├── infra/                # Cloud launch scripts
├── tests/                # Test suite (pytest)
├── docs/                 # Documentation
└── .github/workflows/    # CI
```

---

## Key Commands

| Task | Command |
|------|---------|
| Pretrain (10M) | `python train/pretrain.py --config configs/gpt2-10m.yaml` |
| SFT | `python train/sft.py --base-checkpoint checkpoints/...` |
| DPO | `python train/dpo.py --sft-checkpoint checkpoints/...` |
| Evaluate | `python eval/benchmark.py --checkpoint checkpoints/...` |
| Generate | `python inference/generate.py --checkpoint checkpoints/...` |
| API Server | `python inference/api.py --checkpoint checkpoints/...` |
| Run tests | `pytest tests/` |
| Lint | `ruff check .` |
| Type check | `mypy bharat/` |
| Calculate params | `python scripts/calculate_params.py --all --weight-dtype bf16` |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `WANDB_API_KEY` | Recommended | Experiment tracking |
| `HF_TOKEN` | Optional | Gated HF datasets, model push |

---

## License

MIT — see [LICENSE](LICENSE) for details.
