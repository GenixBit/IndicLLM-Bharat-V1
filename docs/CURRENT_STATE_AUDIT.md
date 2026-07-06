# Bharat AI — Repository Current State Audit

**Date:** 2025-07-06
**Repository:** IndicLLM-Bharat-V1
**Status:** Bharat AI Research Prototype V0

---

## 1. Executive Summary

The repository is a GPT-2-style decoder-only language model prototype focused on Indic languages. It implements a full but minimal pipeline: data preparation → pretraining → SFT → DPO → evaluation → inference → export.

### What Works (Verified)
- GPT-2 model forward/backward pass (train/pretrain.py)
- Single-GPU pretraining with cosine LR schedule, gradient accumulation, W&B logging
- DDP multi-GPU pretraining (train/pretrain_ddp.py)
- Basic SFT fine-tuning with checkpoint save/load
- Basic DPO with policy/reference model setup
- Perplexity, token accuracy, and sample generation evaluation
- Interactive generation REPL
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints
- GGUF export pipeline (HF → llama.cpp → GGUF → Ollama)
- FineWeb-Edu English data pipeline
- Indic data pipeline (Sangraha, Wikipedia, CulturaX, IndicCorp, mC4)
- HuggingFace Hub push script
- AWS GPU instance launcher and teardown
- Environment sanity check (nanoGPT on Shakespeare)
- W&B integration

### What Is Incomplete
- **Tokenizer inconsistency**: GPT-2 tokenizer hardcoded in SFT, DPO, inference, API; SentencePiece used in Indic pipeline
- **SFT loss masking**: All tokens contribute to loss — user, system, and padding tokens are not masked
- **DPO batch-level prompt length**: Single `prompt_len` used for entire batch (line 153 of dpo.py)
- **uint16 token storage**: Prevents vocabularies > 65,535 IDs
- **Checkpoint metadata**: No tokenizer info, no data version, no git SHA in checkpoints
- **Checkpoint resume**: Incomplete — no scheduler state, no random state recovery
- **Evaluation**: Only perplexity + token accuracy + sample generation; no standard benchmarks wired
- **API**: No streaming support, no authentication, wildcard CORS
- **Tests**: Zero test files exist

### What Is Experimental
- `configs/gpt2-1b.yaml`: Unrealistic cost estimate ($20k for 400k iters on 8×A100)
- `configs/gpt2-350m.yaml`: Cost estimates likely underestimates
- DDP training: Uses `GradScaler` with bfloat16 (not needed — bf16 has native grad scaling)
- `prepare_indic.py`: Trains SentencePiece tokenizer but SFT/DPO/inference all use GPT-2 tokenizer

### What Is Broken or Missing
- **No tests**: Zero test files across the entire repository
- **No CI**: No GitHub Actions or CI configuration
- **No streaming API**: `/v1/chat/completions` does not support `stream: true`
- **No authentication**: API has no auth middleware
- **No rate limiting**: API has no rate limiting
- **Wildcard CORS**: `allow_origins=["*"]` in production default
- **No type checking**: No mypy/pyright configuration
- **No linting**: No ruff/black configuration
- **No pre-commit hooks**: No pre-commit configuration
- **No safety documentation**: No model cards, safety evaluation, or governance docs
- **No data manifests**: No versioned, checksummed dataset manifests
- **No contamination checking**: No benchmark contamination detection
- **No PII filtering**: No PII detection or removal in data pipeline
- **No deduplication**: No exact or fuzzy deduplication in data pipeline
- **No licensing validation**: No systematic licence checking for data sources
- **Tokenizer metadata not in checkpoints**: Tokenizer type, hash, special tokens not stored
- **SFT special tokens**: `tokenizer.add_special_tokens` changes embedding size but model was initialized with original vocab_size
- **SFT dataset**: Hardcoded prompt template, no system/user/assistant roles, no multi-turn
- **DPO logprob function**: `prompt_len[0].item()` uses first sample's prompt length for entire batch
- **sentencepiece not in requirements.txt**: Required by `prepare_indic.py` but not listed

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

The repository has **three inconsistent tokenizer paths**:

| Pipeline | Tokenizer | Where |
|----------|-----------|-------|
| English data prep | GPT-2 (HF) or custom BPE | `data/prepare_data.py:84-87` |
| Indic data prep | SentencePiece or GPT-2 | `data/prepare_indic.py:244-288` |
| Pretraining | Reads `vocab_size` from `meta.pkl` | `train/pretrain.py:204-208` |
| SFT | GPT-2 + custom special tokens | `train/sft.py:73-79` |
| DPO | GPT-2 + custom special tokens | `train/dpo.py:128-130` |
| Evaluation | GPT-2 | `eval/benchmark.py:235-236` |
| Inference | GPT-2 | `inference/generate.py:76-80` |
| API | GPT-2 | `inference/api.py:261-262` |
| Export | GPT-2 | `inference/export_ollama.py:125-126` |
| Hub push | GPT-2 or SentencePiece/LLaMA | `scripts/push_to_hub.py:171-188` |

**Problems:**
1. SFT adds `["<|instruction|>", "<|response|>"]` to tokenizer but model was initialized with original vocab_size — embedding mismatch
2. Indic pipeline trains SentencePiece but other pipelines use GPT-2
3. No tokenizer metadata stored in checkpoints
4. No tokenizer compatibility validation on checkpoint load
5. `uint16` storage limits vocab to 65,535

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
2. Hardcode GPT-2 tokenizer + custom special tokens
3. Load JSONL instruction/response pairs
4. Freeze embeddings
5. Train with same forward/backward as pretraining
6. Save checkpoint + best.pt

**DPO** (`train/dpo.py`):
1. Load SFT checkpoint as policy + reference
2. Freeze reference model
3. Load JSONL preference pairs
4. For each batch: compute chosen/rejected logprobs
5. DPO loss: `-log(sigmoid(beta * (chosen_ratio - rejected_ratio)))`
6. Track reward accuracy
7. Save checkpoint

### 2.5 Checkpoint Format

Checkpoints are Python dicts saved with `torch.save()`:

```python
{
    "model": model.state_dict(),      # Full model weights
    "optimizer": optimizer.state_dict(),  # Optimizer state (sometimes)
    "iter_num": int,                   # Training iteration
    "config": dict,                    # Full YAML config
    # Missing:
    # - tokenizer_type, tokenizer_hash, vocab_size
    # - git commit SHA
    # - data version
    # - training seed
    # - package versions
    # - scheduler state (only in some checkpoints)
    # - random state
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
2. Load GPT-2 tokenizer
3. Encode prompt → generate tokens (top-k + top-p sampling)
4. Decode and display
5. Interactive REPL mode

**api.py (FastAPI):**
1. Load checkpoint at startup
2. Expose `/v1/chat/completions` and `/v1/completions`
3. Simple chat template (System/User/Assistant)
4. Top-p sampling only (no top-k)
5. EOS token IDs hardcoded (50256, 50257)
6. Wildcard CORS

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

## 3. Verified Defect Table

### CRITICAL

| # | File | Function/Class | Evidence | Impact | Fix | Regression Risk | Test Required |
|---|------|---------------|----------|--------|-----|-----------------|---------------|
| C1 | `train/sft.py:55-69` | `SFTDataset.__getitem__` | Template includes instruction+response both as input; `__getitem__` returns full sequence as both x and y — no loss masking | Loss is calculated on user instruction and padding tokens, not just assistant response; model penalized for "correctly" predicting its own input | Implement assistant-only loss masking with `-100` labels for non-assistant tokens | Medium — changes loss values, affects checkpoints | SFT masking unit test |
| C2 | `train/dpo.py:153` | `main()` | `pl = prompt_len[0].item()` uses first sample's prompt length for entire batch | Variable-length prompts are incorrectly masked; chosen/rejected logprobs are computed with wrong mask boundaries | Build per-sample response masks | Medium — affects DPO training correctness | DPO masking unit test |
| C3 | `train/dpo.py:66-77` | `log_probs()` | Uses single `prompt_len` for entire batch; mask shape doesn't account for per-sample variable lengths | All samples in batch must have same prompt length or logprob computation is wrong | Implement per-sample variable-length masks | Medium | DPO variable-length test |
| C4 | `train/sft.py:76-78` | `get_tokenizer()` | Adds special tokens `["<|instruction|>", "<|response|>"]` after model initialization; tokenizer.vocab_size changes but model's `wte` and `lm_head` remain at original size | Embedding dimension mismatch — model can't embed the new tokens correctly; causes silent errors or crashes | Initialize model with correct vocab_size including special tokens, or resize embeddings | High — affects SFT checkpoint compatibility | Tokenizer-model size consistency test |

### HIGH

| # | File | Function/Class | Evidence | Impact | Fix | Regression Risk | Test Required |
|---|------|---------------|----------|--------|-----|-----------------|---------------|
| H1 | `data/prepare_data.py:103` | `encode_texts()` | `dtype=np.uint16` hardcoded | Vocabulary > 65,535 causes overflow and data corruption | Auto-detect: uint16 for <65536, uint32 otherwise | Low — existing shards fit in uint16 | uint16/uint32 storage test |
| H2 | `data/prepare_indic.py:310` | `encode_and_write()` | `dtype=np.uint16` hardcoded | Same as H1 for Indic data | Same as H1 | Low | Same as H1 |
| H3 | `data/download_indic.py:214` | `tokenize_and_write()` | `dtype=np.uint16` hardcoded | Same as H1 for downloaded data | Same as H1 | Low | Same as H1 |
| H4 | `train/pretrain.py:152-155` | `load_bin()` | Casts uint16 → int64 every time; no dtype flexibility | Wastes memory (int64 = 8 bytes vs uint16 = 2 bytes) | Auto-select dtype based on vocab_size | Medium | Storage format test |
| H5 | `inference/generate.py:76-80` | `load_tokenizer()` | GPT-2 tokenizer hardcoded | Can't load models trained with SentencePiece or custom tokenizers | Use unified tokenizer interface | High — affects all inference | Tokenizer compatibility test |
| H6 | `train/sft.py:73-79` | `get_tokenizer()` | GPT-2 tokenizer hardcoded | Same as H5 | Use unified tokenizer interface | High — affects SFT | Tokenizer test |
| H7 | `train/dpo.py:128-130` | `main()` | GPT-2 tokenizer hardcoded | Same as H5 | Use unified tokenizer interface | High — affects DPO | Tokenizer test |
| H8 | `eval/benchmark.py:235-236` | `main()` | GPT-2 tokenizer hardcoded | Same as H5 | Use unified tokenizer interface | Medium | Tokenizer test |
| H9 | `inference/api.py:261-262` | `load_model()` | GPT-2 tokenizer hardcoded | Same as H5 | Use unified tokenizer interface | High — affects API | Tokenizer test |
| H10 | All checkpoints | — | No tokenizer metadata stored | Can't validate tokenizer compatibility on load; mixed tokenizers cause silent corruption | Add tokenizer_type, tokenizer_hash, vocab_size to every checkpoint | Medium | Checkpoint metadata test |
| H11 | `train/pretrain.py:277-283` | `main()` | Checkpoint has no tokenizer info, data version, git SHA, seed | Unreproducible experiments | Add reproducibility metadata | Low | Checkpoint test |
| H12 | `train/pretrain_ddp.py:174-176` | `main()` | Same as H11 | Same | Same | Low | Checkpoint test |
| H13 | `inference/api.py:60-65` | Global | `allow_origins=["*"]` wildcard CORS | Security risk in production | Make CORS configurable, default to specific origins | Low | CORS config test |

### MEDIUM

| # | File | Function/Class | Evidence | Impact | Fix | Regression Risk | Test Required |
|---|------|---------------|----------|--------|-----|-----------------|---------------|
| M1 | `train/pretrain.py:269-283` | `main()` | Checkpoint only saved at eval_interval; no periodic save | If training crashes between eval intervals, up to 500 steps of work lost | Add periodic checkpointing (every N steps) | Low | Checkpoint save test |
| M2 | `train/pretrain_ddp.py:170-178` | `main()` | Same as M1 | Same | Same | Low | Same |
| M3 | `train/pretrain_ddp.py:161` | `main()` | `scaler = torch.cuda.amp.GradScaler()` with bfloat16 | bf16 doesn't need GradScaler; causes unnecessary overhead | Only use GradScaler for fp16 | Low | DDP precision test |
| M4 | `train/pretrain.py` | `main()` | No NaN detection | NaN loss can go undetected until crash | Add NaN detection with save-and-exit | Low | NaN detection test |
| M5 | `train/pretrain_ddp.py` | `main()` | Same as M4 | Same | Same | Low | Same |
| M6 | `train/pretrain.py` | `main()` | No random state in checkpoint | Training can't be resumed with exact reproducibility | Save rng_state in checkpoint | Low | Reproducibility test |
| M7 | `train/pretrain.py` | `main()` | No scheduler state in checkpoint | LR schedule doesn't resume correctly | Save scheduler state | Low | Resume test |
| M8 | `inference/api.py:145-146` | `generate()` | Hardcoded EOS token IDs (50256, 50257) | Models with different tokenizer won't stop correctly | Use tokenizer.eos_token_id | Medium | API stop token test |
| M9 | `inference/api.py` | `chat_completions()` | No streaming support | API not OpenAI-compatible for streaming use cases | Add streaming with SSE | Medium | Streaming test |
| M10 | `inference/api.py` | Global | No authentication | Anyone can call the API | Add API key auth middleware | Low | Auth test |
| M11 | `inference/api.py` | Global | No rate limiting | Unbounded request volume | Add rate limiting | Low | Rate limit test |
| M12 | `inference/api.py:217-231` | `completions()` | Text completions endpoint, but not fully OpenAI-compatible | Partial incompatibility | Match OpenAI spec | Low | API spec test |
| M13 | `inference/api.py:151-162` | `format_chat_prompt()` | Simple template, no function calling support | Can't use modern chat features | Add function calling format | Medium | Function calling test |
| M14 | `requirements.txt` | — | `sentencepiece` missing | `prepare_indic.py` will fail with ImportError | Add to requirements.txt | Low | — |
| M15 | `requirements.txt` | — | `datasketch>=1.6.0` listed but not used anywhere | Unnecessary dependency | Remove | Low | — |
| M16 | `configs/gpt2-1b.yaml:46-49` | — | `model.n_embd: 2048` with `n_head: 16` — d_head = 128 (unusual) | Possibly incorrect architecture — standard is 64 or 96 dim per head | Validate architecture: 2048/16 = 128 (OK but unusual) | Low | — |
| M17 | `configs/gpt2-1b.yaml:49` | — | `estimated_cost_usd: 20000` for 400 hours on 8×A100 | At $3/hr/A100 spot, 400h × 8 × $3 = $9,600; at on-demand $4/hr = $12,800; $20k seems high | Correct or remove specific dollar estimate; provide range | Low | — |

### LOW

| # | File | Function/Class | Evidence | Impact | Fix | Regression Risk | Test Required |
|---|------|---------------|----------|--------|-----|-----------------|---------------|
| L1 | All `.py` files | — | No type hints in many functions | Code harder to maintain and debug | Add type hints | Low | — |
| L2 | All `.py` files | — | No docstrings on some functions | Unclear API | Add docstrings | Low | — |
| L3 | `train/pretrain.py:229` | `main()` | `model.configure_optimizers` check always returns None | Dead code path | Remove or implement | Low | — |
| L4 | `eval/benchmark.py:235-236` | `main()` | GPT-2 tokenizer import inside main; slow | Negligible | Move to top-level import | Low | — |
| L5 | `eval/results_gpt2-10m_*.json` | — | Results commit to git | Benchmark results should be tracked but not in repo root | Move to `eval/results/` | Low | — |
| L6 | `.gitignore` | — | `*.pt`, `*.bin` patterns duplicated | Redundant but harmless | Clean up | Low | — |
| L7 | `train/sft.py:133-135` | `main()` | Embeddings frozen — may not be optimal for all SFT scenarios | Potential underfitting | Make configurable | Medium | — |
| L8 | `inference/export_ollama.py:102` | `convert_to_hf()` | `TRANSPOSE_KEYS` for Conv1D compatibility | Works but duplicates logic in `scripts/push_to_hub.py:153` | Extract shared utility | Low | — |
| L9 | `scripts/push_to_hub.py:153` | `convert_to_hf()` | Same weight transpose logic | Duplicated from export_ollama.py | Share tokenizer/convert utils | Low | — |
| L10 | `infra/aws_launch.sh:88` | — | SSH from anywhere (0.0.0.0/0) | Security risk | Document that user should restrict | Low | — |

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
- [ ] All tests pass (pytest)
- [ ] Unified tokenizer interface used by pretraining, SFT, DPO, eval, inference, API
- [ ] Tokenizer metadata stored in all checkpoints
- [ ] Tokenizer compatibility validated on checkpoint load
- [ ] SFT loss applies only to assistant response tokens (verified by test)
- [ ] DPO uses per-sample response masks (verified by test)
- [ ] Checkpoint resume saves/restores optimizer, scheduler, random state
- [ ] Training is restartable after interruption
- [ ] CPU smoke test completes
- [ ] CI passes (lint, type check, unit tests, tokenizer test, API test)
- [ ] Linting (ruff) and formatting configured
- [ ] No unsupported performance claims in README
- [ ] Roadmap separates completed/active/planned work
