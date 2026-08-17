# Milestone 6.2 — Bharat Architecture Pretraining Engine & Overfit Validation

## Status: Complete ✅

This document defines the pretraining pipeline, optimization strategy, learning rate schedule, loss computation, checkpoint metadata binding, and overfit-on-one-batch validation test suite for the **`BharatForCausalLM`** architecture.

---

## 1. Objectives

1. Replace legacy prototype scripts with a native, modern pretraining loop for `BharatForCausalLM` supporting RoPE, RMSNorm, SwiGLU, and GQA.
2. Implement memory-mapped binary shard reading for scalable offline data streaming.
3. Validate gradient flow and training dynamics via deterministic Overfit-on-One-Batch testing (driving cross-entropy loss from $\approx \ln(V)$ down to $< 0.1$).
4. Implement metadata-rich checkpoint saving (capturing git SHA, tokenizer hash, package versions, and RNG states).
5. Seamlessly connect checkpoints with BharatBench evaluation harness (`LocalCausalLMAdapter`).

---

## 2. Technical Architecture

### 2.1 Optimization Strategy
* **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$).
* **Weight Decay Parameter Partitioning**:
  * 2D weight matrices (attention projections, MLP gate/up/down projections, embeddings): `weight_decay = 0.1`.
  * 1D parameters (RMSNorm scales, biases): `weight_decay = 0.0`.
* **Gradient Clipping**: Maximum gradient norm threshold $= 1.0$.
* **Mixed Precision**: Automatic Mixed Precision (AMP) supporting `bfloat16`, `float16`, and `float32`.

### 2.2 Learning Rate Schedule
Cosine learning rate decay with linear warmup:
$$\eta_t = \begin{cases} 
\eta_{\max} \cdot \frac{t + 1}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right) & \text{if } T_{\text{warmup}} \le t < T_{\max} \\
\eta_{\min} & \text{if } t \ge T_{\max}
\end{cases}$$
Where $\eta_{\min} = 0.1 \times \eta_{\max}$.

### 2.3 Checkpoint Contract
Checkpoints saved via `bharat.training.checkpointing.save_checkpoint` record:
```json
{
  "model": "<state_dict>",
  "optimizer": "<optimizer_state_dict>",
  "config": { ... },
  "metadata": {
    "git_sha": "<40-char SHA>",
    "tokenizer_type": "<bpe | gpt2 | sp>",
    "tokenizer_hash": "<sha256>",
    "vocab_size": 64000,
    "torch_version": "...",
    "training_step": 1000
  },
  "rng_state": { ... }
}
```

---

## 3. CLI Usage

### Pretraining Command
```bash
python scripts/pretrain_bharat.py \
  --config configs/models/bharat-350m.yaml \
  --data data/shards/train_0000.bin \
  --val-data data/shards/val_0000.bin \
  --batch-size 4 \
  --seq-len 4096 \
  --grad-accum 8 \
  --lr 3e-4 \
  --warmup-iters 200 \
  --max-iters 10000 \
  --device cuda \
  --dtype bfloat16 \
  --output-dir checkpoints/bharat-350m
```

### Synthetic Smoke / Validation Run
```bash
python scripts/pretrain_bharat.py \
  --config configs/models/bharat-350m.yaml \
  --max-iters 10 \
  --batch-size 2 \
  --seq-len 128 \
  --synthetic-data \
  --device cpu \
  --dtype float32 \
  --json
```

---

## 4. Verification Evidence

### Overfit-on-One-Batch Proof
* **Initial Loss**: $\approx 5.54$ ($\approx \ln(256)$)
* **Final Loss**: $< 0.08$ ($< 0.10$ acceptance threshold)
* **Status**: Passed in `tests/training/test_overfit_350m.py`.

### Post-Training Integration Proof
* SFT instruction tuning with assistant-only loss masking: Verified (`tests/posttraining/test_modern_posttraining.py`).
* DPO reference-policy preference training: Verified.
* Local evaluation adapter (`LocalCausalLMAdapter`) evaluation: Verified (`tests/eval/test_local_inference_checkpoint.py`).
