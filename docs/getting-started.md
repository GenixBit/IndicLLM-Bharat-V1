# Getting Started with Bharat AI

Step-by-step guide to go from zero to a trained foundation model.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 or 3.12 | 3.14 has ML lib gaps; avoid |
| Git | any | |
| Ollama | any | optional, for local inference testing |
| Cloud account | — | RunPod or Lambda Labs |

---

## Phase 0 — Environment Setup

```bash
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git
cd IndicLLM-Bharat-V1

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set secrets (optional)
cp .env.example .env
# Edit .env: add WANDB_API_KEY, HF_TOKEN, etc.
```

---

## Phase 1 — Sanity Check (GPT-2 10M Pretrain)

Goal: confirm PyTorch, MPS/CUDA, and the training pipeline work end-to-end.

```bash
source .venv/bin/activate
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 500
```

Expected output:

```
step 0: train loss 4.2xxx, val loss 4.2xxx
iter 0: loss 4.2xxx, time ...ms
...
iter 490: loss 1.5xxx, time ...ms
step 500: train loss 1.5xxx, val loss 1.6xxx
```

- Runtime: ~5 min on M2 Mac, ~15 min on CPU
- Loss should drop from ~4.2 → ~1.5 over 500 iters
- Checkpoint saved to `checkpoints/gpt2-10m/`

---

## Phase 2 — Data Pipeline (FineWeb-Edu)

Goal: produce `data/shards/train.bin` and `data/shards/val.bin`.

```bash
# Quick sanity run (~30 sec, ~1M tokens)
python data/prepare_data.py --subset sample-10BT --max-docs 200

# Enough for a real smoke-test (~5M tokens, ~2 min)
python data/prepare_data.py --subset sample-10BT --max-docs 2000

# Full 10B-token subset for cloud training (run overnight or on cloud)
python data/prepare_data.py --subset sample-10BT

# Optional: custom BPE tokenizer for 350M+ models
python data/prepare_data.py --subset sample-10BT --train-tokenizer --vocab-size 32000
```

Outputs:

```
data/shards/
├── train.bin      # memory-mapped uint16 token array
├── val.bin        # 1% held-out
├── meta.pkl       # vocab_size, token counts
└── DATASET.md     # dataset card (sources, filters, token counts)
```

**Note:** `gpt2-10m.yaml` and `gpt2-124m.yaml` use GPT-2's tokenizer (vocab 50257).
`gpt2-350m.yaml` and `gpt2-1b.yaml` use a custom 32k BPE tokenizer — run with `--train-tokenizer`.

---

## Phase 3 — Local Smoke-Test (10M Model, ~10 min on M2)

```bash
# Full 2000-iter run — confirms loss convergence
python train/pretrain.py --config configs/gpt2-10m.yaml

# 20-iter sanity check — just confirms no import/shape errors
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 20
```

Expected: loss drops from ~10 → ~3–4 over 2000 iters on real data.
If loss stays flat or explodes, debug data prep before going to cloud.

---

## Phase 4 — Cloud Pretraining (124M on 1× A100)

| Provider | Strength | A100 80GB spot price |
|----------|----------|----------------------|
| RunPod | Easiest, web UI, Network Volumes | ~$1.50–$2/hr |
| Lambda Labs | More stable, persistent filesystems | ~$1.50–$2/hr |
| AWS p4d | Best for multi-node, harder setup | ~$3–4/hr |

```bash
# Review checklist, generates bootstrap script
CONFIG=configs/gpt2-124m.yaml bash infra/runpod_launch.sh

# Or for Lambda Labs
CONFIG=configs/gpt2-124m.yaml bash infra/lambda_launch.sh
```

Then SSH into the pod and run:

```bash
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git llm-lab && cd llm-lab
export WANDB_API_KEY=<your-key>
bash infra/runpod_bootstrap.sh     # or lambda_bootstrap.sh
```

Expected training time: ~24–48h for 100k iters on 1× A100 80GB.
Expected final loss: ~2.8–3.2 (GPT-2 quality range).

---

## Phase 5 — Eval and W&B Tracking

```bash
export WANDB_API_KEY=your_key

# Run evaluation benchmarks
python eval/benchmark.py --checkpoint checkpoints/gpt2-124m/ckpt.pt
```

Key metrics to track:
- `val/loss` — should be decreasing and close to `train/loss`
- HellaSwag accuracy — random baseline is 25%; GPT-2 124M gets ~29–31%
- Tokens/sec — use this to estimate final cost before launching longer runs

---

## Phase 6 — Alignment (SFT + DPO)

```bash
# Stage 1: Supervised fine-tuning on instruction/response pairs
python train/sft.py \
  --base-checkpoint checkpoints/gpt2-124m/final.pt \
  --output checkpoints/gpt2-124m-sft \
  --dataset teknium/OpenHermes-2.5 \
  --max-samples 10000

# Stage 2: DPO preference tuning
python train/dpo.py \
  --model checkpoints/gpt2-124m-sft \
  --output checkpoints/gpt2-124m-dpo
```

> **Note:** The legacy SFT/DPO scripts are being replaced by `bharat/posttraining/` with proper loss masking and per-sample handling. See [docs/ROADMAP.md](ROADMAP.md).

---

## Phase 7 — Inference and API

### Local Inference via Ollama

```bash
python inference/export_ollama.py \
  --model checkpoints/gpt2-124m-sft \
  --name bharat-124m

ollama create bharat-124m -f inference/ollama/Modelfile
ollama run bharat-124m
```

### OpenAI-Compatible REST API

```bash
MODEL_PATH=checkpoints/gpt2-124m-sft \
API_KEY=your-secret-key \
python inference/api.py
```

Test:

```bash
curl http://localhost:8000/health

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is a transformer?"}],
    "max_tokens": 200
  }'
```

---

## Cost Tracking

| Date | Run | Model | Provider | GPUs | Hours | $/hr | Total $ | Tokens | $/1B tok | Notes |
|------|-----|-------|----------|------|-------|------|---------|--------|----------|-------|
|      |     |       |          |      |       |      |         |        |          |       |

**Target:** under $0.10 per billion tokens trained.
**Rule:** never launch a run > $50 without a passing 20-iter smoke-test first.

---

## Scaling Path

Once 124M is working and loss curves look clean:

1. Run `prepare_data.py` with full `sample-10BT` (no `--max-docs`)
2. Switch to `configs/gpt2-350m.yaml` — uses custom tokenizer, needs 2× A100
3. Then `configs/gpt2-1b.yaml` — needs 8× A100, ~$20k; worth raising budget or cloud credits first

For multi-GPU: `pretrain.py` currently runs single-GPU. Multi-GPU support planned for the modern `bharat/` model (Milestone 2).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `loss = nan` from iter 0 | Bad data shards or LR too high | Check `meta.pkl` token count; reduce LR |
| Loss flat after warmup | Data too small for model size | Use more docs in `prepare_data.py` |
| `MPS` OOM on 10M model | `block_size` too large | Reduce `block_size` or `--max-iters 20` |
| `WANDB_API_KEY not set` | `.env` not loaded | `source .venv/bin/activate && export $(cat .env \| xargs)` |
| `meta.pkl` missing | `prepare_data.py` not run yet | Run data prep first |
