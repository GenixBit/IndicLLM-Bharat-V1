# llm-lab

Foundation model training lab: data pipeline → pretraining → alignment → eval → inference.

Built to take a 10M smoke-test all the way to a 1B+ parameter model you fully own.

---

## Quick start

```bash
# 1. Setup environment (Python 3.11 or 3.12 recommended)
bash scripts/setup.sh
source .venv/bin/activate

# 2. Local sanity check — nanoGPT on Shakespeare (~5 min on M2, ~15 min on CPU)
bash scripts/run_local_sanity.sh

# 3. Smoke-test the full pipeline on your Mac before paying for cloud
python data/prepare_data.py --subset sample-10BT --max-docs 200
python train/pretrain.py --config configs/gpt2-10m.yaml

# 4. Cloud training (A100)
CONFIG=configs/gpt2-124m.yaml bash infra/runpod_launch.sh
CONFIG=configs/gpt2-124m.yaml bash infra/lambda_launch.sh
```

---

## Project layout

```
configs/     YAML model + training configs (10M, 124M, 350M, 1B)
data/        FineWeb-Edu download, clean, tokenize, shard
train/       pretrain.py, sft.py, dpo.py, utils.py
eval/        run_eval.py  — lm-eval-harness + W&B logging
inference/   api.py (OpenAI-compatible), export_ollama.py (GGUF)
infra/       runpod_launch.sh, lambda_launch.sh + bootstrap scripts
scripts/     setup.sh, run_local_sanity.sh
docs/        getting-started.md, run logs, cost tracking
```

---

## Model configs

| Config | Params | GPU | Est. cost | Use case |
|--------|--------|-----|-----------|----------|
| `gpt2-10m.yaml` | ~10M | M2 / CPU | $0 | Local smoke-test, pipeline validation |
| `gpt2-124m.yaml` | ~124M | 1× A100 | ~$150 | Learning, first cloud run |
| `gpt2-350m.yaml` | ~350M | 2× A100 | ~$3k | MVP foundation model |
| `gpt2-1b.yaml` | ~1B | 8× A100 | ~$20k | Serious v1 |

Always run `gpt2-10m` locally first to confirm your data pipeline is correct before paying for cloud time.

---

## Step-by-step workflow

### Phase 0 — Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
cp .env.example .env   # then fill in WANDB_API_KEY, HF_TOKEN etc.
```

### Phase 1 — Local sanity (nanoGPT on Shakespeare)

```bash
bash scripts/run_local_sanity.sh
# Expected output: 500 training iterations, loss ~1.4–1.6, no errors
```

### Phase 2 — Data pipeline

```bash
# Quick dry-run (~30 sec, 200 docs, ~1M tokens) — do this first
python data/prepare_data.py --subset sample-10BT --max-docs 200

# Larger dry-run (enough to train 10M for a real convergence test)
python data/prepare_data.py --subset sample-10BT --max-docs 5000

# Full 10B-token subset (long download, run on cloud or overnight)
python data/prepare_data.py --subset sample-10BT

# Custom BPE tokenizer (for 350M+ models — vocab_size=32000)
python data/prepare_data.py --subset sample-10BT --train-tokenizer --vocab-size 32000
```

Output: `data/shards/train.bin`, `data/shards/val.bin`, `data/shards/meta.pkl`, `data/shards/DATASET.md`

### Phase 3 — Local smoke-test (10M, ~10 min on M2)

```bash
# After prepare_data.py with at least --max-docs 200
python train/pretrain.py --config configs/gpt2-10m.yaml

# Single quick pass to check for import / shape errors only
python train/pretrain.py --config configs/gpt2-10m.yaml --max-iters 20
```

Confirm: loss decreases, checkpoint saved to `checkpoints/gpt2-10m/`.

### Phase 4 — Cloud pretraining (124M)

```bash
# Review checklist and generate bootstrap script
CONFIG=configs/gpt2-124m.yaml bash infra/runpod_launch.sh
# or
CONFIG=configs/gpt2-124m.yaml bash infra/lambda_launch.sh

# Then on the cloud pod:
bash infra/runpod_bootstrap.sh   # or lambda_bootstrap.sh
```

### Phase 5 — Eval

```bash
export WANDB_API_KEY=your_key
python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt
python eval/run_eval.py --checkpoint checkpoints/gpt2-124m/ckpt.pt --tasks hellaswag,piqa,winogrande
```

Results written to `eval/results.json` and logged to W&B.

### Phase 6 — Alignment (SFT + DPO)

```bash
# Supervised fine-tuning on instruction data
python train/sft.py --base-checkpoint checkpoints/gpt2-124m/final.pt \
                    --output checkpoints/gpt2-124m-sft

# DPO preference tuning
python train/dpo.py --model checkpoints/gpt2-124m-sft \
                    --output checkpoints/gpt2-124m-dpo
```

### Phase 7 — Inference

```bash
# Local inference via Ollama
python inference/export_ollama.py --model checkpoints/gpt2-124m-sft --name llm-lab-124m
ollama create llm-lab-124m -f inference/ollama/Modelfile
ollama run llm-lab-124m

# OpenAI-compatible API (run manually in a separate terminal)
MODEL_PATH=checkpoints/gpt2-124m-sft uvicorn inference.api:app --host 0.0.0.0 --port 8000

# Test the API
curl http://localhost:8000/health
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Environment variables

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Purpose |
|----------|----------|---------|
| `WANDB_API_KEY` | Recommended | Experiment tracking |
| `HF_TOKEN` | Optional | Gated HF datasets |
| `RUNPOD_API_KEY` | Optional | RunPod cloud launches |
| `LAMBDA_API_KEY` | Optional | Lambda Labs launches |
| `API_KEY` | For inference | Auth for `/v1/chat/completions` |
| `MODEL_PATH` | For inference | Path to SFT/DPO checkpoint |

---

## Cost tracking

Log every cloud run here. Target: <$0.10 per billion tokens.

| Date | Run name | Model | GPU | Hours | $/hr | Total $ | Tokens | $/1B tok |
|------|----------|-------|-----|-------|------|---------|--------|----------|
|      |          |       |     |       |      |         |        |          |

Runs are also logged automatically to W&B if `WANDB_API_KEY` is set.

---

## Reference

- [nanoGPT](https://github.com/karpathy/nanoGPT) — local sanity runs
- [LitGPT](https://github.com/Lightning-AI/litgpt) — next step up from nanoGPT
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — training data
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) — evals
- [TRL](https://github.com/huggingface/trl) — SFT + DPO
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — tokens ≈ 20× params
- [OLMo](https://github.com/allenai/OLMo), [TinyLlama](https://github.com/jzhang38/TinyLlama) — open pretrain references
