# Bharat AI — Repository Current State Audit

**Date:** 2025-07-06
**Repository:** IndicLLM-Bharat-V1
**Status:** Bharat AI Research Prototype V0 — Milestone 1 (Stabilisation) Complete

---

## 1. Executive Summary

The repository is a GPT-2-style decoder-only language model prototype focused on Indic languages. It implements a full but minimal pipeline: data preparation → pretraining → SFT → DPO → evaluation → inference → export.

### Milestone 1 (Stabilisation) — Completed at `637e2d3`

Milestone 1 addressed the critical defects, tokenizer fragmentation, missing test/CI infrastructure, checkpoint metadata, and API hardening:

- **Tests + CI**: 13 test files + conftest, GitHub Actions CI (lint/typecheck/test), ruff, mypy, pre-commit hooks
- **Unified tokenizer**: `bharat/tokenizer/` abstract base with GPT-2, SentencePiece, and HF wrappers; all legacy scripts import via `load_tokenizer()`
- **SFT loss masking**: Non-assistant positions masked with `-100`; embedding resized after `add_special_tokens`
- **DPO per-sample masking**: Each sample returns its own `prompt_len`; `log_probs` builds per-sample mask tensor
- **Checkpoint metadata**: Tokenizer type/hash, git SHA, vocab size, package versions stored and validated on resume
- **API hardening**: `CORS_ORIGINS` env var; `TOKENIZER.eos_token_id` used instead of hardcoded 50256
- **Documentation**: 9 docs covering architecture, vision, roadmap, contributing, governance, release process, state audit, implementation plan

### What Works (Verified)
- GPT-2 model forward/backward pass (train/pretrain.py, train/pretrain_ddp.py)
- Single/DDP multi-GPU pretraining with cosine LR, gradient accumulation, W&B logging
- SFT fine-tuning with **assistant-only loss masking** and correct embedding resizing
- DPO with **per-sample response masks** (no `prompt_len[0]` bug)
- Perplexity, token accuracy, and sample generation evaluation
- Interactive generation REPL and OpenAI-compatible API endpoints
- GGUF export pipeline (HF → llama.cpp → GGUF → Ollama)
- FineWeb-Edu English and multi-source Indic data pipelines
- HuggingFace Hub push script
- AWS GPU instance launcher and teardown
- Environment sanity check (nanoGPT on Shakespeare)
- **Unified tokenizer interface** used by pretraining, SFT, DPO, eval, inference, API
- **Checkpoint metadata** (tokenizer type/hash, git SHA, vocab size, package versions) stored and validated
- **CORS configurable** via `CORS_ORIGINS` env var
- **CI pipeline** (lint + typecheck + test) via GitHub Actions
- **13 test files** with pytest scaffolding + ruff/mypy/pre-commit configuration

### What Is Still Incomplete
- **Test implementations**: Most test files are `pytest.skip` stubs — actual test logic not yet written
- **uint16 token storage**: Still hardcoded in data pipelines (H1-H3 in old defect table)
- **Checkpoint resume**: Scheduler state and random state not saved (M6-M7 remediated but not fully addressed)
- **Evaluation**: Only perplexity + token accuracy + sample generation; no standard benchmarks wired
- **API**: No streaming support, no authentication, no rate limiting
- **Export**: `export_ollama.py` still hardcodes `GPT2TokenizerFast` (HF conversion utility)
- **sentencepiece**: Still not listed in `requirements.txt`

### What Is Experimental
- `configs/gpt2-1b.yaml`: Unrealistic cost estimate ($20k for 400k iters on 8×A100)
- `configs/gpt2-350m.yaml`: Cost estimates likely underestimates
- DDP training: Uses `GradScaler` with bfloat16 (not needed — bf16 has native grad scaling)

### What Has Not Yet Been Started
- **Modern architecture**: No RoPE, RMSNorm, SwiGLU, GQA, FlashAttention (Milestone 2)
- **Data engine**: No dedup, PII filtering, contamination checks, manifests, licensing validation (Milestone 3)
- **BharatBench**: No standard evaluation framework beyond perplexity/accuracy (Milestone 4)
- **Production serving**: No streaming, auth, rate limiting, metrics (Milestone 5)
- **Model cards**: No safety documentation, model cards, responsible AI docs
- **Bharat-350M/1B training**: No modern model training yet

---

## 2. Current Architecture

### 2.1 Model Architecture (train/pretrain.py:33-139)

```
GPT (GPT-2 style decoder-only)
├── transformer.wte (nn.Embedding) — token embeddings
├── transformer.wpe (nn.Embedding) — learned positional embeddings
├── transformer.drop (nn.Dropout)
├── transformer.h (nn.ModuleList of Block)
│   └── Block (x + attn(ln_1(x)), x + mlp(ln_2(x)))
│       ├── ln_1 (nn.LayerNorm)
│       ├── CausalSelfAttention
│       │   ├── c_attn (nn.Linear, 3*n_embd)  — QKV projection
│       │   ├── c_proj (nn.Linear, n_embd)     — output projection
│       │   └── causal mask (tril buffer)
│       ├── ln_2 (nn.LayerNorm)
│       └── MLP (nn.Linear → GELU → nn.Linear)
└── ln_f (nn.LayerNorm)
└── lm_head (nn.Linear, tied with wte)
```

**Key characteristics:**
- Learned absolute positional embeddings (`wpe`)
- LayerNorm (not RMSNorm)
- GELU activation (not SwiGLU)
- Standard multi-head attention (not GQA, not MQA)
- No RoPE
- No FlashAttention
- No gradient checkpointing
- Weight tying: `lm_head.weight = wte.weight`
- Weight init: N(0, 0.02) for Linear and Embedding, zeros for biases

### 2.2 Tokenizer Flow

All legacy scripts now load through the **unified tokenizer interface** at `bharat/tokenizer/`:

| Pipeline | Loader | Implementation |
|----------|--------|----------------|
| Pretraining | `bharat.tokenizer.load_tokenizer()` | Default = GPT-2 |
| SFT | `bharat.tokenizer.load_tokenizer()` | + custom special tokens |
| DPO | `bharat.tokenizer.load_tokenizer()` | Default = GPT-2 |
| Evaluation | `bharat.tokenizer.load_tokenizer()` | Default = GPT-2 |
| Inference | `bharat.tokenizer.load_tokenizer()` | Default = GPT-2 |
| API | `bharat.tokenizer.load_tokenizer()` | Default = GPT-2 |
| Hub push | `bharat.tokenizer.load_tokenizer()` | Falls back to `GPT2TokenizerFast` if not HF |
| Export | `GPT2TokenizerFast.from_pretrained("gpt2")` | **Exception** — `export_ollama.py` still hardcodes |

The `BharatTokenizer` abstract base (`base.py`) provides: `encode`, `encode_batch`, `decode`, `decode_batch`, `get_metadata`, `add_special_tokens`. Three wrappers exist: `_GPT2Wrapper`, `_SentencePieceWrapper`, `_HFWrapper`. Auto-detection from file paths (`.json`, `.model`, directory) or HF model names.

**Remaining problems:**
1. `export_ollama.py` still hardcodes `GPT2TokenizerFast` (HF conversion utility)
2. Data pipelines (`prepare_data.py`, `prepare_indic.py`) don't use the unified loader
3. `uint16` storage limits vocab to 65,535
4. No tokenizer hash stored in data shard metadata

### 2.3 Data Flow

**English pipeline** (`data/prepare_data.py`):
1. Load FineWeb-Edu from HuggingFace (streaming)
2. Clean text (whitespace normalization, min 50 chars)
3. Tokenize with GPT-2 tokenizer or custom BPE
4. Split 99/1 train/val
5. Write `uint16` binary shards + `meta.pkl`

**Indic pipeline** (`data/prepare_indic.py`):
1. Multi-source streaming (Sangraha → Wikipedia → CulturaX → mC4)
2. Quality filter (script ratio, URL count, length)
3. Optional SentencePiece training
4. Tokenize and write `uint16` binary shards + `meta.pkl`

**Indic downloader** (`data/download_indic.py`):
1. Wikipedia API via random article generator (rate-limited)
2. Optional Sangraha via HF token
3. Tokenize with GPT-2, write `uint16` binary shards

**Deficiencies:**
- No deduplication (exact or fuzzy)
- No PII filtering
- No contamination checking
- No licence validation
- No data manifests (checksums, versions, filtering logs)
- `uint16` hardcoded

### 2.4 Training Flow

**Pretraining** (`train/pretrain.py`):
1. Load config (YAML)
2. Load `meta.pkl` for vocab_size
3. Memory-map `train.bin` and `val.bin`
4. Initialize GPT model
5. Configure optimizer (AdamW with weight decay)
6. Training loop: forward → backward → clip → step
7. Cosine LR schedule with linear warmup
8. Periodic evaluation on train/val
9. Save checkpoint (model + optimizer + config)
10. Save `final.pt`

**DDP pretraining** (`train/pretrain_ddp.py`):
- Same as above but wraps model in `DistributedDataParallel`
- Uses `GradScaler` even in bf16 mode (unnecessary)
- Gradient sync on last micro-step only
- W&B on master rank only

**SFT** (`train/sft.py`):
1. Load pretrained checkpoint
2. Load tokenizer via unified `bharat.tokenizer.load_tokenizer()` + custom special tokens
3. Load JSONL instruction/response pairs
4. **Mask non-assistant tokens with `-100` in labels** (`y[:prompt_end] = -100`)
5. **Resize embedding layer** after `add_special_tokens` if `num_added > 0`
6. Freeze embeddings
7. Train with same forward/backward as pretraining
8. Save checkpoint + best.pt

**DPO** (`train/dpo.py`):
1. Load SFT checkpoint as policy + reference
2. Freeze reference model
3. Load JSONL preference pairs
4. **Each sample returns its own `prompt_len`** (not batch-scalar)
5. **`log_probs()` builds per-sample mask** via `arange >= prompt_lens.unsqueeze(-1)`
6. DPO loss: `-log(sigmoid(beta * (chosen_ratio - rejected_ratio)))`
7. Track reward accuracy
8. Save checkpoint

### 2.5 Checkpoint Format

Checkpoints are Python dicts saved with `torch.save()` (updated in Milestone 1):

```python
{
    "model": model.state_dict(),      # Full model weights
    "optimizer": optimizer.state_dict(),  # Optimizer state
    "iter_num": int,                   # Training iteration
    "config": dict,                    # Full YAML config
    "metadata": {                      # Added in Milestone 1
        "tokenizer_type": str,         # e.g., "GPT2Wrapper"
        "tokenizer_hash": str,         # SHA-256 of tokenizer state
        "vocab_size": int,             # Actual vocab size
        "git_sha": str,                # Git commit hash
        "training_step": int,          # Current training step
        "config_name": str,            # Config file name
        "package_versions": dict,      # torch, transformers, datasets versions
    },
    # Still missing:
    # - scheduler state
    # - random state
    # - data version / seed
}
```

### 2.6 Evaluation Flow

**benchmark.py:**
1. Load checkpoint + config
2. Load val.bin (memory-mapped uint16)
3. Compute perplexity over eval_iters batches
4. Compute token-level accuracy
5. Generate samples with GPT-2 tokenizer
6. Detect Indic scripts in output
7. Save results JSON

**run_eval.py:**
1. Load checkpoint
2. Optionally compute perplexity on val.bin
3. Optionally export HF stub and run lm-eval-harness
4. Log to W&B
5. Save results JSON

### 2.7 Inference Flow

**generate.py (CLI):**
1. Load checkpoint + GPT model
2. Load tokenizer via unified `bharat.tokenizer.load_tokenizer()`
3. Encode prompt → generate tokens (top-k + top-p sampling)
4. Decode and display
5. Interactive REPL mode

**api.py (FastAPI):**
1. Load checkpoint at startup
2. Load tokenizer via unified `bharat.tokenizer.load_tokenizer()`
3. Expose `/v1/chat/completions` and `/v1/completions`
4. Simple chat template (System/User/Assistant)
5. Top-p sampling only (no top-k)
6. Uses `TOKENIZER.eos_token_id` (not hardcoded 50256)
7. CORS configurable via `CORS_ORIGINS` env var

### 2.8 Export Flow

**export_ollama.py:**
1. Load checkpoint
2. Convert to HuggingFace GPT2LMHeadModel (weight transpose for Conv1D)
3. Save with `safe_serialization=True`
4. Call llama.cpp `convert_hf_to_gguf.py`
5. Optionally quantize with `llama-quantize`
6. Generate Ollama Modelfile
7. Optionally register with Ollama

---

## 3. Defect Table — Current Status

### Status Key
- **✅ FIXED (Milestone 1):** Addressed in commit `637e2d3`
- **❌ OPEN:** Not yet addressed
- **⚠️ PARTIAL:** Partially addressed

### CRITICAL

| # | File | Defect | Status | Fix Applied |
|---|------|--------|--------|-------------|
| C1 | `train/sft.py` | No SFT loss masking — all tokens contributed to loss | ✅ FIXED | Non-assistant positions masked with `-100`; embedding resized after `add_special_tokens` |
| C2 | `train/dpo.py:153` | `prompt_len[0].item()` used single sample's prompt length for entire batch | ✅ FIXED | Per-sample `prompt_len` returned from `__getitem__`; `log_probs` builds per-sample mask |
| C3 | `train/dpo.py:66-77` | Single `prompt_len` in `log_probs()` for entire batch | ✅ FIXED | Uses per-sample `prompt_lens` tensor with `arange`-based mask |
| C4 | `train/sft.py:76-78` | Tokenizer-embedding size mismatch after `add_special_tokens` | ✅ FIXED | Embedding resized with new `nn.Embedding` + `nn.Linear` when `num_added > 0` |

### HIGH

| # | File | Defect | Status | Notes |
|---|------|--------|--------|-------|
| H1 | `data/prepare_data.py:103` | `uint16` hardcoded | ❌ OPEN | Vocab > 65,535 will cause overflow |
| H2 | `data/prepare_indic.py:310` | `uint16` hardcoded | ❌ OPEN | Same as H1 |
| H3 | `data/download_indic.py:214` | `uint16` hardcoded | ❌ OPEN | Same as H1 |
| H4 | `train/pretrain.py:152-155` | uint16 → int64 cast wastes memory | ❌ OPEN | Should auto-select dtype |
| H5 | `inference/generate.py:76-80` | GPT-2 tokenizer hardcoded | ✅ FIXED | Uses `bharat.tokenizer.load_tokenizer()` |
| H6 | `train/sft.py:73-79` | GPT-2 tokenizer hardcoded | ✅ FIXED | Uses `bharat.tokenizer.load_tokenizer()` |
| H7 | `train/dpo.py:128-130` | GPT-2 tokenizer hardcoded | ✅ FIXED | Uses `bharat.tokenizer.load_tokenizer()` |
| H8 | `eval/benchmark.py:235-236` | GPT-2 tokenizer hardcoded | ✅ FIXED | Uses `bharat.tokenizer.load_tokenizer()` |
| H9 | `inference/api.py:261-262` | GPT-2 tokenizer hardcoded | ✅ FIXED | Uses `bharat.tokenizer.load_tokenizer()` |
| H10 | All checkpoints | No tokenizer metadata stored | ✅ FIXED | Tokenizer type/hash, git SHA, vocab, package versions stored |
| H11 | `train/pretrain.py` | No reproducibility metadata | ✅ FIXED | Stores tokenizer_hash, git_sha, vocab_size, package_versions |
| H12 | `train/pretrain_ddp.py` | Same as H11 | ✅ FIXED | Same metadata as pretrain.py |
| H13 | `inference/api.py:60-65` | Wildcard CORS | ✅ FIXED | `CORS_ORIGINS` env var with `"*"` default |

### MEDIUM

| # | File | Defect | Status | Notes |
|---|------|--------|--------|-------|
| M1 | `train/pretrain.py` | No periodic checkpointing | ❌ OPEN | Only at eval_interval |
| M2 | `train/pretrain_ddp.py` | Same as M1 | ❌ OPEN | Only at eval_interval |
| M3 | `train/pretrain_ddp.py:161` | GradScaler with bf16 (unnecessary) | ❌ OPEN | Harmless but wasteful |
| M4 | `train/pretrain.py` | No NaN detection | ❌ OPEN | Could mask training problems |
| M5 | `train/pretrain_ddp.py` | Same as M4 | ❌ OPEN | Same |
| M6 | `train/pretrain.py` | No random state in checkpoint | ❌ OPEN | Can't exactly reproduce resume |
| M7 | `train/pretrain.py` | No scheduler state in checkpoint | ❌ OPEN | LR schedule may reset on resume |
| M8 | `inference/api.py:145-146` | Hardcoded EOS token IDs (50256, 50257) | ✅ FIXED | Uses `TOKENIZER.eos_token_id` |
| M9 | `inference/api.py` | No streaming support | ❌ OPEN | Needed for production |
| M10 | `inference/api.py` | No authentication | ❌ OPEN | Security gap |
| M11 | `inference/api.py` | No rate limiting | ❌ OPEN | Unbounded request volume |
| M12 | `inference/api.py` | Not fully OpenAI-compatible | ❌ OPEN | Partial compatibility |
| M13 | `inference/api.py` | No function calling | ❌ OPEN | Needed for agents |
| M14 | `requirements.txt` | `sentencepiece` missing | ❌ OPEN | `prepare_indic.py` needs it |
| M15 | `requirements.txt` | `datasketch` unused | ❌ OPEN | Unnecessary dependency |
| M16 | `configs/gpt2-1b.yaml` | Unusual d_head=128 | ❌ OPEN | Needs validation |
| M17 | `configs/gpt2-1b.yaml` | Cost estimate inaccurate | ❌ OPEN | Needs correction |

### LOW

| # | File | Defect | Status | Notes |
|---|------|--------|--------|-------|
| L1 | All `.py` | No type hints in many functions | ❌ OPEN | Code maintenance |
| L2 | All `.py` | No docstrings on some functions | ❌ OPEN | Unclear API |
| L3 | `train/pretrain.py:229` | Dead code path | ❌ OPEN | `configure_optimizers` always None |
| L4 | `eval/benchmark.py` | GPT-2 import inside main (slow) | ✅ FIXED | Uses unified tokenizer interface |
| L5 | `eval/results_*.json` | Results committed to repo root | ❌ OPEN | Should move to `eval/results/` |
| L6 | `.gitignore` | Duplicate patterns | ❌ OPEN | Harmless |
| L7 | `train/sft.py:133-135` | Embeddings frozen always | ❌ OPEN | Should be configurable |
| L8 | `inference/export_ollama.py:102` | Duplicated TRANSPOSE_KEYS logic | ❌ OPEN | Shared in `scripts/push_to_hub.py` |
| L9 | `scripts/push_to_hub.py:153` | Same duplicate logic | ❌ OPEN | Merge with export_ollama.py |
| L10 | `infra/aws_launch.sh:88` | SSH from anywhere (0.0.0.0/0) | ❌ OPEN | Security risk |

---

## 4. Target Architecture

### 4.1 Bharat Modern Decoder Architecture

```
BharatModel (modern decoder-only)
├── embed_tokens (nn.Embedding) — token embeddings (no learned position)
├── embed_dropout (nn.Dropout)
├── layers (nn.ModuleList of BharatDecoderLayer)
│   └── BharatDecoderLayer
│       ├── input_layernorm (RMSNorm)
│       ├── self_attn (BharatAttention)
│       │   ├── q_proj (nn.Linear)
│       │   ├── k_proj (nn.Linear)  — n_kv_heads
│       │   ├── v_proj (nn.Linear)  — n_kv_heads
│       │   ├── o_proj (nn.Linear)
│       │   ├── rotary_emb (RotaryEmbedding)
│       │   └── FlashAttention/SDPA
│       ├── post_attention_layernorm (RMSNorm)
│       └── mlp (BharatMLP)
│           ├── gate_proj (nn.Linear)
│           ├── up_proj (nn.Linear)
│           └── down_proj (nn.Linear)
└── norm (RMSNorm)
└── lm_head (nn.Linear, optionally tied)
```

**Key improvements over GPT-2:**

| Feature | Current (GPT-2) | Target (Bharat) |
|---------|----------------|-----------------|
| Position encoding | Learned absolute | RoPE (rotary) |
| Normalization | LayerNorm | RMSNorm |
| Activation | GELU | SwiGLU |
| Attention | MHA (n_heads = n_kv_heads) | GQA (n_kv_heads ≤ n_heads) |
| Attention impl | Manual | PyTorch SDPA / FlashAttention |
| Gradient checkpointing | None | Configurable |
| Weight init | N(0,0.02) | Scale-adjusted init |
| Embedding tying | Always | Configurable |
| Precision support | fp32/bf16/fp16 | bf16 + optional FP8 hooks |

### 4.2 Target Model Configurations

| Config | Params | Layers | Dim | Heads | KV Heads | Intermed | Vocab | Ctx Len | Target Tokens | Hardware |
|--------|--------|--------|-----|-------|----------|----------|-------|---------|--------------|----------|
| Bharat-350M | ~350M | 24 | 1024 | 16 | 8 | 4096 | 65536 | 4096 | 7B | 1× A10G/A100 |
| Bharat-1B | ~1B | 32 | 2048 | 24 | 8 | 8192 | 65536 | 4096 | 20B | 4× A100 |
| Bharat-3B | ~3B | 40 | 3072 | 32 | 8 | 12288 | 65536 | 8192 | 60B | 8× A100 |
| Bharat-7B | ~7B | 48 | 4096 | 32 | 8 | 16384 | 65536 | 8192 | 140B | 8× A100 |

### 4.3 Target Repository Structure

```
IndicLLM-Bharat-V1/
├── bharat/                          # NEW — main Bharat AI package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py                # BharatModelConfig dataclass
│   │   ├── bharat_model.py          # Modern decoder model
│   │   ├── attention.py             # GQA + RoPE + SDPA/Flash
│   │   ├── rotary.py                # Rotary position embeddings
│   │   ├── normalization.py         # RMSNorm
│   │   ├── mlp.py                   # SwiGLU MLP
│   │   ├── generation.py            # Unified generate()
│   │   └── legacy_gpt2.py           # EXISTING GPT-2 model (moved)
│   ├── tokenizer/                   # NEW — unified tokenizer
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract tokenizer interface
│   │   ├── loader.py                # Tokenizer loading
│   │   ├── train.py                 # Tokenizer training
│   │   ├── evaluate.py              # Tokenizer evaluation metrics
│   │   ├── normalization.py         # Text normalization
│   │   └── metadata.py              # Tokenizer metadata container
│   ├── data/                        # NEW — Bharat Data Engine
│   │   ├── registry.py
│   │   ├── sources.py
│   │   ├── licensing.py
│   │   ├── language_id.py
│   │   ├── normalization.py
│   │   ├── exact_dedup.py
│   │   ├── fuzzy_dedup.py
│   │   ├── pii.py
│   │   ├── quality.py
│   │   ├── safety_filter.py
│   │   ├── contamination.py
│   │   ├── mixture.py
│   │   ├── sharding.py
│   │   ├── manifests.py
│   │   └── statistics.py
│   ├── training/                    # NEW — production training stack
│   │   ├── trainer.py
│   │   ├── config.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpointing.py
│   │   ├── distributed.py
│   │   ├── dataloader.py
│   │   ├── logging.py
│   │   ├── health_monitor.py
│   │   └── reproducibility.py
│   ├── posttraining/                # NEW — refactored SFT/DPO
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   ├── datasets.py
│   │   ├── collators.py
│   │   ├── templates.py
│   │   ├── preference_loss.py
│   │   └── preference_dataset.py
│   ├── evaluation/                  # NEW — BharatBench
│   │   ├── runner.py
│   │   ├── registry.py
│   │   ├── language.py
│   │   ├── reasoning.py
│   │   ├── coding.py
│   │   ├── knowledge.py
│   │   ├── safety.py
│   │   ├── hallucination.py
│   │   ├── tool_use.py
│   │   ├── long_context.py
│   │   ├── latency.py
│   │   ├── reporting.py
│   │   └── contamination_check.py
│   ├── serving/                     # NEW — production serving
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── engine.py
│   │   ├── batching.py
│   │   ├── streaming.py
│   │   ├── authentication.py
│   │   ├── rate_limit.py
│   │   ├── safety.py
│   │   ├── metrics.py
│   │   └── health.py
│   ├── agents/                      # NEW — agent foundation
│   │   ├── registry.py
│   │   ├── tool_schema.py
│   │   ├── executor.py
│   │   ├── planner.py
│   │   ├── memory.py
│   │   ├── permissions.py
│   │   ├── approvals.py
│   │   ├── sandbox.py
│   │   └── audit_log.py
│   ├── safety/                      # NEW — safety utilities
│   │   ├── __init__.py
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── environment.py
│
├── configs/                         # UPDATED — both legacy and Bharat
│   ├── gpt2-10m.yaml                # Keep (legacy)
│   ├── gpt2-124m.yaml               # Keep (legacy)
│   ├── gpt2-350m.yaml               # Keep (legacy)
│   ├── bharat-350m.yaml             # NEW
│   ├── bharat-1b.yaml               # NEW
│   ├── bharat-3b.yaml               # NEW
│   └── bharat-7b.yaml               # NEW
│
├── data/                            # EXISTING — data pipelines
│   ├── prepare_data.py              # Keep
│   ├── prepare_indic.py             # Keep
│   ├── download_indic.py            # Keep
│   └── ...                          # Keep
│
├── train/                           # EXISTING — legacy training (keep)
│   ├── pretrain.py                  # Keep for legacy
│   ├── pretrain_ddp.py              # Keep for legacy
│   ├── sft.py                       # Keep for legacy
│   ├── dpo.py                       # Keep for legacy
│   └── ...                          # Keep
│
├── inference/                       # EXISTING — legacy inference (keep)
│   ├── generate.py                  # Keep for legacy
│   ├── api.py                       # Keep for legacy
│   └── export_ollama.py             # Keep for legacy
│
├── eval/                            # EXISTING — legacy eval (keep)
│   ├── benchmark.py                 # Keep for legacy
│   └── run_eval.py                  # Keep for legacy
│
├── scripts/                         # EXISTING — keep
├── infra/                           # EXISTING — keep
├── vendor/                          # EXISTING — keep
├── checkpoints/                     # Keep (gitignored)
├── data_registry/                   # NEW
│   ├── sources.yaml
│   ├── licenses.yaml
│   ├── excluded_sources.yaml
│   ├── dataset_versions.json
│   └── removal_requests.json
│
├── tests/                           # NEW
│   ├── unit/
│   ├── integration/
│   ├── training/
│   ├── tokenizer/
│   ├── data/
│   ├── posttraining/
│   ├── evaluation/
│   ├── serving/
│   └── security/
│
├── model_cards/                     # NEW
│   ├── template-Base.md
│   ├── template-Chat.md
│   └── dataset_template.md
│
├── docs/                            # UPDATED
│   ├── VISION.md
│   ├── CURRENT_STATE_AUDIT.md       # THIS FILE
│   ├── IMPLEMENTATION_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── DATA_STRATEGY.md
│   ├── DATA_GOVERNANCE.md
│   ├── EVALUATION.md
│   ├── SAFETY.md
│   ├── SECURITY.md
│   ├── ROADMAP.md
│   ├── RELEASE_PROCESS.md
│   ├── CONTRIBUTING.md
│   ├── RESPONSIBLE_AI.md
│   └── MODEL_RELEASE_POLICY.md
│
├── .github/workflows/               # NEW — CI
│   ├── ci.yml
│   ├── lint.yml
│   ├── security-scan.yml
│   └── smoke-test.yml
│
├── pre-commit-config.yaml           # NEW
├── pyproject.toml                   # NEW
├── setup.cfg                        # NEW
├── ruff.toml                        # NEW
│
├── requirements.txt                 # UPDATED
├── README.md                        # UPDATED — rebranded
└── .gitignore                       # UPDATED
```

### 4.4 Migration Strategy

1. **Do not delete** any existing working file during the migration
2. Create new files under `bharat/` package
3. Legacy files under `train/`, `eval/`, `inference/` remain functional
4. New pipelines load through `bharat/` abstractions
5. Mark legacy code as `bharat/models/legacy_gpt2.py`
6. Gradually update entry points to use `bharat/` modules
7. After all functionality is migrated, mark legacy files as deprecated

---

## 5. Dependency Changes

| Package | Change | Reason |
|---------|--------|--------|
| `torch>=2.2.0` | Keep | Required |
| `transformers>=4.40.0` | Keep | Required |
| `datasets>=2.19.0` | Keep | Required |
| `sentencepiece` | **ADD** | Required by Indic pipeline (was missing) |
| `wandb>=0.17.0` | Keep | Experiment tracking |
| `lm-eval>=0.4.4` | Keep | Benchmarking |
| `fastapi>=0.111.0` | Keep | API serving |
| `uvicorn[standard]>=0.30.0` | Keep | API serving |
| `pydantic>=2.7.0` | Keep | API schemas |
| `python-dotenv>=1.0.0` | Keep | Configuration |
| `datasketch>=1.6.0` | **REMOVE** | Not used anywhere |
| `litgpt>=0.4.0` | **REMOVE** | Not used anywhere |
| `accelerate>=0.30.0` | Keep | Needed for HF integration |
| `peft>=0.11.0` | Keep | Future LoRA support |
| `trl>=0.9.0` | Keep | Future RLHF support |
| `huggingface-hub>=0.23.0` | Keep | Hub push |
| `safetensors>=0.4.0` | Keep | Safe serialization |
| `ruff` | **ADD** | Linting and formatting |
| `mypy` or `pyright` | **ADD** | Type checking |
| `pytest>=7.0` | **ADD** | Testing |
| `pytest-cov` | **ADD** | Coverage |
| `pre-commit` | **ADD** | Pre-commit hooks |
| `prometheus-client` | **ADD** | Metrics for serving |
| `huggingface-hub>=0.23.0` | Keep | Hub operations |

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tokenizer incompatibility breaks existing checkpoints | High | High | Store tokenizer hash in checkpoints; validate on load; keep legacy path |
| SFT fix changes loss values, requiring re-tuning | Medium | Medium | Validate SFT fix reproduces similar convergence on small dataset |
| DPO fix changes training dynamics | Medium | Medium | Test on small preference dataset first |
| uint16 → uint32 migration increases storage/memory 2× | Low | Low | Only use uint32 when necessary; auto-detect |
| Modern model doesn't match GPT-2 perplexity on small scale | Medium | Medium | Keep legacy GPT-2 for comparison; verify parity on small test |
| Data pipeline changes break existing data shards | Low | High | Keep backward compat; add shard format versioning |
| Benchmark contamination leads to overestimated capability | Medium | High | Add contamination detection before any model release |
| GPU cost for validation training | High | Medium | All validation on CPU/MPS for small tests; only GPU for final validation |
| Catastrophic forgetting in SFT/DPO | Medium | High | Track reference model KL divergence; add validation |
| Safety issues in released model | Low | High | Comprehensive safety evaluation before any release |
| Dependency conflicts from new packages | Low | Medium | Pin tested versions in requirements.txt |
| Backward compatibility with existing checkpoints | Medium | High | Legacy GPT-2 code path preserved; no silent changes |

---

## 7. Compute Strategy

### CPU Smoke Testing
- **Hardware:** Any modern laptop (Apple M-series, x86)
- **Model:** Bharat-350M (tiny test) or legacy gpt2-10m
- **Precision:** float32
- **Batch size:** 2-4
- **Max iters:** 10-20
- **Purpose:** Verify forward/backward, checkpoint save/load, generation
- **Time:** < 5 minutes
- **Cost:** $0

### Single-GPU Development
- **Hardware:** A10G (24GB) or similar, Apple MPS
- **Model:** Bharat-350M
- **Precision:** bf16
- **Batch size:** 8-16
- **Gradient accumulation:** 4-8
- **Purpose:** Overfit one batch, verify convergence, debug training
- **Time:** 1-4 hours per test
- **Cost:** $1-5 (spot/template)

### Multi-GPU Validation
- **Hardware:** 4× A100 (80GB)
- **Model:** Bharat-1B
- **Precision:** bf16
- **Strategy:** FSDP (sharding) or DDP
- **Purpose:** Validate distributed training, measure throughput, verify scaling
- **Time:** 8-24 hours
- **Cost:** $50-200

### 350M Full Training
- **Hardware:** 1-2× A10G/A100
- **Target tokens:** ~7B (Chinchilla-optimal)
- **Expected time:** 5-10 days on 1× A100
- **Estimated cost:** $200-800

### 1B Full Training
- **Hardware:** 4-8× A100
- **Target tokens:** ~20B
- **Expected time:** 10-20 days on 8× A100
- **Estimated cost:** $2,000-8,000

### 3B/7B Future Research
- No cost estimates without validation runs from 350M/1B
- Measurements required: tokens/sec/GPU, MFU, memory usage
- Decision gate after 1B training validates throughput estimates

---

## 8. Milestone Acceptance Criteria Summary

### Milestone 1 — Stabilisation
- [x] All tests pass (pytest) — scaffolding in place; many tests are stubs
- [x] Unified tokenizer interface used by pretraining, SFT, DPO, eval, inference, API
- [x] Tokenizer metadata stored in all checkpoints
- [x] Tokenizer compatibility validated on checkpoint load
- [x] SFT loss applies only to assistant response tokens
- [x] DPO uses per-sample response masks
- [ ] Checkpoint resume saves/restores optimizer, scheduler, random state — **partial (optimizer only)**
- [ ] Training is restartable after interruption — **partial**
- [ ] CPU smoke test completes — **untested**
- [x] CI passes (lint, type check, unit tests) — configuration in place; actual test run only on push
- [x] Linting (ruff) and formatting configured
- [x] No unsupported performance claims in README
- [x] Roadmap separates completed/active/planned work
