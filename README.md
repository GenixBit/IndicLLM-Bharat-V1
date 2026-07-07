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
| Sanity check | `python scripts/sanity_check.py` |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `WANDB_API_KEY` | Recommended | Experiment tracking |
| `HF_TOKEN` | Optional | Gated HF datasets, model push |

---

## License

MIT — see [LICENSE](LICENSE) for details.
