# Getting started

Step-by-step guide to go from zero to a trained foundation model.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11 or 3.12 | 3.14 has ML lib gaps; avoid |
| Git | any | for cloning nanoGPT into vendor/ |
| Ollama | any | optional, for local inference testing |
| Cloud account | — | RunPod or Lambda Labs |

---

## Phase 0 — Environment setup

```bash
# Clone (or cd into) the repo
cd ~/Projects/llm-lab

# Create venv, install deps, clone nanoGPT
bash scripts/setup.sh

# Activate for all subsequent commands
source .venv/bin/activate

# Set secrets
cp .env.example .env
# Edit .env: add WANDB_API_KEY, HF_TOKEN, etc.
```

What `setup.sh` does:
- Creates `.venv` with Python 3.12 (or 3.11)
- Installs PyTorch (MPS build on Mac, CUDA on Linux)
- Installs all `requirements.txt` packages
- Clones `karpathy/nanoGPT` into `vendor/nanoGPT`

---

## Phase 1 — Local sanity check (nanoGPT on Shakespeare)

Goal: confirm PyTorch + MPS + nanoGPT work end-to-end on your Mac before touching cloud.

```bash
bash scripts/run_local_sanity.sh
```

Expected output:
```
step 0: train loss 4.2xxx, val loss 4.2xxx
iter 0: loss 4.2xxx, time ...ms
...
iter 490: loss 1.5xxx, time ...ms
step 500: train loss 1.5xxx, val loss 1.6xxx
Local sanity check passed.
```

- Runtime: ~5 min on M2, ~15 min on CPU
- Loss should drop from ~4.2 → ~1.5 over 500 iters
- If you see CUDA errors: the script falls back to CPU automatically

---

## Phase 2 — Data pipeline (FineWeb-Edu)

Goal: produce `data/shards/train.bin` and `data/shards/val.bin` ready for `pretrain.py`.

```bash
# Step 1: quick sanity run (~30 sec, ~1M tokens)
python data/prepare_data.py --subset sample-10BT --max-docs 200

# Step 2: enough data for a real local smoke-test (~5M tokens, ~2 min)
python data/prepare_data.py --subset sample-10BT --max-docs 2000

# Step 3: full 10B-token subset for cloud training (long — run overnight or on cloud)
python data/prepare_data.py --subset sample-10BT

# Step 4 (optional): custom BPE tokenizer for 350M+ models
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

The `gpt2-10m.yaml` and `gpt2-124m.yaml` configs use GPT-2's tokenizer (vocab 50257).
The `gpt2-350m.yaml` and `gpt2-1b.yaml` configs use a custom 32k BPE tokenizer — run with `--train-tokenizer`.

---

## Phase 3 — Local smoke-test (10M model, ~10 min on M2)

Goal: confirm your data shards and `pretrain.py` work correctly before launching a cloud job.

```bash
# Full 2000-iter run — confirms loss convergence
python train/pretrain.py --config configs/gpt2-10m.yaml

# 20-iter sanity check — just confirms no import/shape errors
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 20
```

Expected: loss drops from ~10 → ~3–4 over 2000 iters on real data. If loss stays flat or explodes, debug data prep before going to cloud.

Checkpoint saved to: `checkpoints/gpt2-10m/`

---

## Phase 4 — Cloud pretraining (124M on 1× A100)

### Choose a provider

| Provider | Strength | A100 80GB spot price |
|----------|----------|----------------------|
| RunPod | Easiest, web UI, Network Volumes | ~$1.50–$2/hr |
| Lambda Labs | More stable, persistent filesystems | ~$1.50–$2/hr |
| AWS p4d | Best for multi-node, harder setup | ~$3–4/hr |

Start with RunPod or Lambda. Both use the same bootstrap script.

```bash
# Review checklist, generates infra/runpod_bootstrap.sh
CONFIG=configs/gpt2-124m.yaml bash infra/runpod_launch.sh

# Or for Lambda Labs
CONFIG=configs/gpt2-124m.yaml bash infra/lambda_launch.sh
```

Then SSH into the pod and run:
```bash
git clone <your-repo-url> llm-lab && cd llm-lab
export WANDB_API_KEY=<your-key>
bash infra/runpod_bootstrap.sh     # or lambda_bootstrap.sh
```

Expected training time: ~24–48h for 100k iters on 1× A100 80GB.
Expected final loss: ~2.8–3.2 (GPT-2 quality range).

### Checkpoint survival

- RunPod: mount a Network Volume at `/workspace`; checkpoints survive pod stops
- Lambda: attach a persistent filesystem; same idea
- Both: `eval_interval` in the config controls how often `ckpt.pt` is written (default: every 500 iters)

---

## Phase 5 — Eval and W&B tracking

```bash
export WANDB_API_KEY=your_key

# Run lm-eval-harness benchmarks
python eval/run_eval.py \
  --checkpoint checkpoints/gpt2-124m/ckpt.pt \
  --tasks hellaswag,piqa,winogrande

# Results are written to eval/results.json and logged to W&B
```

Key metrics to track:
- `val/loss` — should be decreasing and close to `train/loss`
- HellaSwag accuracy — random baseline is 25%; GPT-2 124M gets ~29–31%
- Tokens/sec — use this to estimate final cost before launching longer runs

W&B setup: add `WANDB_API_KEY` to `.env`. The `wandb.enabled: true` flag in each config controls logging.

---

## Phase 6 — Alignment (SFT + DPO)

Raw pretrained models generate text but don't follow instructions. Add two fine-tuning stages:

```bash
# Stage 1: Supervised fine-tuning on instruction/response pairs
python train/sft.py \
  --base-checkpoint checkpoints/gpt2-124m/final.pt \
  --output checkpoints/gpt2-124m-sft \
  --dataset teknium/OpenHermes-2.5 \
  --max-samples 10000

# Stage 2: DPO preference tuning (makes the model more helpful / less harmful)
python train/dpo.py \
  --model checkpoints/gpt2-124m-sft \
  --output checkpoints/gpt2-124m-dpo
```

This is the minimum to produce a chat model you can demo.

---

## Phase 7 — Inference and API

### Local inference via Ollama

```bash
# Generate Modelfile
python inference/export_ollama.py \
  --model checkpoints/gpt2-124m-sft \
  --name llm-lab-124m

# Create and run the model
ollama create llm-lab-124m -f inference/ollama/Modelfile
ollama run llm-lab-124m
```

### OpenAI-compatible REST API

Run in a separate terminal:
```bash
MODEL_PATH=checkpoints/gpt2-124m-sft \
API_KEY=your-secret-key \
uvicorn inference.api:app --host 0.0.0.0 --port 8000
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

## Cost tracking

| Date | Run | Model | Provider | GPUs | Hours | $/hr | Total $ | Tokens | $/1B tok | Notes |
|------|-----|-------|----------|------|-------|------|---------|--------|----------|-------|
|      |     |       |          |      |       |      |         |        |          |       |

**Target**: under $0.10 per billion tokens trained.
**Rule**: never launch a run > $50 without a passing 20-iter smoke-test first.

---

## Scaling path

Once 124M is working and loss curves look clean:

1. Run `prepare_data.py` with full `sample-10BT` (no `--max-docs`)
2. Switch to `configs/gpt2-350m.yaml` — uses custom tokenizer, needs 2× A100
3. Then `configs/gpt2-1b.yaml` — needs 8× A100, ~$20k; worth raising budget or cloud credits first

For multi-GPU: `pretrain.py` currently runs single-GPU. Before 350M+ on multiple GPUs, switch to [LitGPT](https://github.com/Lightning-AI/litgpt) or add `torch.distributed` / FSDP to `pretrain.py`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `loss = nan` from iter 0 | Bad data shards or LR too high | Check `meta.pkl` token count; reduce LR |
| Loss flat after warmup | Data too small for model size | Use more docs in `prepare_data.py` |
| `MPS` OOM on 10M model | `block_size` too large | Reduce `block_size` in config or `--max-iters 20` |
| `WANDB_API_KEY not set` | `.env` not loaded | `source .venv/bin/activate` then `export $(cat .env \| xargs)` |
| nanoGPT clone fails | No internet or git not installed | `bash scripts/setup.sh` again |
| `meta.pkl` missing | `prepare_data.py` not run yet | Run data prep first |
