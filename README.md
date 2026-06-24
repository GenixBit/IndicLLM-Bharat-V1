# 🇮🇳 IndicLLM-Bharat-V1

**Open-source multilingual Indian language model — from scratch.**

Train, align, and deploy a foundation model for 13 Indic languages: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Assamese, Urdu, and Sanskrit.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org)

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git
cd IndicLLM-Bharat-V1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (English — FineWeb-Edu)
python data/prepare_data.py --subset sample-10BT --max-docs 5000

# 3. Train 10M model locally (~15 min on CPU)
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 500

# 4. Run benchmark
python eval/benchmark.py --checkpoint checkpoints/gpt2-10m/ckpt.pt

# 5. Interactive generation
python inference/generate.py --checkpoint checkpoints/gpt2-10m/ckpt.pt
```

---

## 📊 Results — 10M Model (Training Complete ✅)

| Metric | Iter 1000 | Iter 2000 (Final) | Δ |
|--------|-----------|-------------------|---|
| **Val Perplexity** | 240.36 | **167.00** | −30% |
| **Val Accuracy** | 19.16% | **22.55%** | +3.4 pts |
| **Train Perplexity** | 214.99 | **144.12** | −33% |
| **Overfitting** | gap=0.112 | gap=0.147 | ✓ Healthy |
| **Gen Speed** | 46 tok/s | **64 tok/s** | +39% |

| Detail | Value |
|--------|-------|
| **Parameters** | 30.1M (6L / 384d / 6H) |
| **Training data** | FineWeb-Edu 51M tokens |
| **Training time** | ~10 hours on AWS c5.2xlarge |
| **Checkpoint** | `checkpoints/gpt2-10m/final.pt` (121MB) |


<details>
<summary>Training loss curve</summary>

```
10.88 ┤●  random init
 7.49 ┤ ●
 5.88 ┤   ●  step 600
 5.53 ┤     ●  step 1000
 5.25 ┤       ●  step 1400
 5.21 ┤        ●  step 1600
      └────────────────────
      0   600  1000  1400  1800  2000
```
</details>

---

## 🏗️ Architecture

```
IndicLLM-Bharat-V1/
├── configs/                          # Model configurations
│   ├── gpt2-10m.yaml                 # 10M — local smoke test
│   ├── gpt2-124m.yaml                # 124M — first cloud run
│   ├── gpt2-124m-indic.yaml          # 124M — Indic-focused
│   └── gpt2-350m.yaml                # 350M — serious training
│
├── data/                             # Data pipelines
│   ├── prepare_data.py               # FineWeb-Edu → binary shards
│   ├── prepare_indic.py              # Indic multilingual pipeline (13 langs)
│   ├── download_indic.py             # Wikipedia API downloader (rate-limited)
│   └── indic/                        # Indic language shards
│
├── train/                            # Training scripts
│   ├── pretrain.py                   # Single-GPU pretraining (GPT-2 arch)
│   ├── pretrain_ddp.py               # Multi-GPU DDP training
│   ├── sft.py                        # Supervised fine-tuning
│   ├── dpo.py                        # Direct preference optimization
│   └── utils.py                      # Config, W&B, checkpoint utils
│
├── inference/                        # Inference & serving
│   ├── generate.py                   # Interactive REPL + CLI generation
│   ├── api.py                        # OpenAI-compatible FastAPI server
│   └── export_ollama.py              # GGUF export + Ollama integration
│
├── eval/                             # Evaluation
│   └── benchmark.py                  # Perplexity, accuracy, sample eval
│
├── scripts/                          # Utilities
│   └── push_to_hub.py                # HuggingFace Hub push
│
└── infra/                            # Cloud infrastructure
    ├── launch_aws.sh                 # AWS GPU instance provisioner
    ├── teardown_aws.sh               # Clean shutdown + cost tracking
    └── ddp_launch.sh                 # Auto single/multi-GPU launcher
```

---

## 🗺️ Training Roadmap

| Stage | Model | GPU | Data | Status |
|-------|-------|-----|------|--------|
| **Stage 1** | 10M | CPU / c5.2xlarge | FineWeb-Edu 51M tok | 🟢 Training |
| **Stage 2** | 124M | A10G (24GB) | FineWeb-Edu 2.5B tok | ⏳ Pending |
| **Stage 3** | 124M-Indic | A10G (24GB) | Indic Wiki + Sangraha | ⏳ Pending |
| **Stage 4** | 350M | A100 (80GB) | FineWeb + Indic 10B tok | 📋 Planned |
| **Stage 5** | SFT | A10G | Instruction datasets | 📋 Planned |
| **Stage 6** | DPO | A10G | Preference data | 📋 Planned |

---

## 🌏 Indic Language Data Pipeline

Supports 13 Indian languages via Wikipedia API + [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) corpus:

```bash
# Download Wikipedia articles for 5 languages (rate-limited, polite crawl)
python data/download_indic.py --langs hi,bn,ta,te,mr --max-articles 500

# Full pipeline: tokenize + filter + shard
python data/prepare_indic.py --source wikimedia --langs all
```

| Language | Script | Wikipedia | Status |
|----------|--------|-----------|--------|
| Hindi (hi) | Devanagari | ✅ 28 articles | Working |
| Bengali (bn) | Bengali | ✅ 30 articles | Working |
| Tamil (ta) | Tamil | ✅ 30 articles | Working |
| Telugu (te) | Telugu | ✅ 27 articles | Working |
| Marathi (mr) | Devanagari | ✅ 28 articles | Working |
| Gujarati (gu) | Gujarati | — | Ready |
| Kannada (kn) | Kannada | — | Ready |
| Malayalam (ml) | Malayalam | — | Ready |
| Odia (or) | Odia | — | Ready |
| Punjabi (pa) | Gurmukhi | — | Ready |
| Assamese (as) | Assamese | — | Ready |
| Urdu (ur) | Arabic | — | Ready |
| Sanskrit (sa) | Devanagari | — | Ready |

---

## 🎯 Model Configurations

| Config | Params | Layers | Dim | Heads | Context | Use Case |
|--------|--------|--------|-----|-------|---------|----------|
| `gpt2-10m` | 30M | 6 | 384 | 6 | 512 | Local smoke test |
| `gpt2-124m` | 124M | 12 | 768 | 12 | 1024 | First cloud run |
| `gpt2-124m-indic` | 124M | 12 | 768 | 12 | 1024 | Indic-focused |
| `gpt2-350m` | 350M | 24 | 1024 | 16 | 1024 | MVP foundation model |

---

## 🔧 Full Pipeline

### 1. Data Preparation

```bash
# English (FineWeb-Edu)
python data/prepare_data.py --subset sample-10BT --max-docs 5000

# Indic (Wikipedia + Sangraha)
python data/download_indic.py --langs hi,bn,ta,te,mr --max-articles 500
```

### 2. Pretraining

```bash
# Local (CPU/MPS)
python train/pretrain.py --config configs/gpt2-10m.yaml

# Cloud single-GPU
python train/pretrain.py --config configs/gpt2-124m.yaml

# Cloud multi-GPU (DDP)
bash infra/ddp_launch.sh configs/gpt2-350m.yaml
```

### 3. Alignment

```bash
# Supervised fine-tuning
python train/sft.py --base-checkpoint checkpoints/gpt2-124m/final.pt

# Direct preference optimization
python train/dpo.py --model checkpoints/gpt2-124m-sft/final.pt
```

### 4. Evaluation

```bash
python eval/benchmark.py --checkpoint checkpoints/gpt2-10m/ckpt.pt
```

### 5. Inference

```bash
# Interactive CLI
python inference/generate.py --checkpoint checkpoints/gpt2-10m/ckpt.pt

# OpenAI-compatible API server
python inference/api.py --checkpoint checkpoints/gpt2-10m/ckpt.pt --port 8000

# Export to GGUF + Ollama
python inference/export_ollama.py --checkpoint checkpoints/gpt2-10m/ckpt.pt --name indicllm-10m
```

### 6. Push to HuggingFace

```bash
python scripts/push_to_hub.py --checkpoint checkpoints/gpt2-10m/ckpt.pt --repo GenixBit/IndicLLM-Bharat
```

---

## ☁️ Cloud Training (AWS)

```bash
# Launch GPU instance
bash infra/launch_aws.sh

# SSH into instance
ssh -i ~/.ssh/indicllm-key.pem ubuntu@<instance-ip>

# Start training
cd ~/IndicLLM-Bharat-V1
source .venv/bin/activate
nohup python train/pretrain.py --config configs/gpt2-124m.yaml > ~/train.log 2>&1 &

# Monitor
tail -f ~/train.log

# Clean shutdown (saves costs)
bash infra/teardown_aws.sh
```

---

## 📦 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `WANDB_API_KEY` | Recommended | Experiment tracking |
| `HF_TOKEN` | Optional | Gated HF datasets, model push |

---

## 🔗 References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Training architecture foundation
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — English training data
- [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) — Indic language corpus
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) — Indian language AI research
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — Optimal tokens ≈ 20× params
- [OLMo](https://github.com/allenai/OLMo), [TinyLlama](https://github.com/jzhang38/TinyLlama) — Open pretrain references

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for Bharat by <a href="https://github.com/GenixBit">GenixBit</a>
</p>
