# Bharat AI Architecture

## Current (Legacy)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  data/      │────▶│  train/      │────▶│  checkpoint │
│  prepare_*.py│     │  pretrain.py │     │  .pt        │
│             │     │  sft.py      │     └─────────────┘
│  FineWeb-Edu│     │  dpo.py      │           │
│  Indic Wiki │     │  utils.py    │           ▼
└─────────────┘     └──────────────┘     ┌─────────────┐
                                         │  inference/ │
                                         │  generate   │
                                         │  api.py     │
                                         │  export*    │
                                         └─────────────┘
```

**Limitations:**
- GPT-2 hardcoded tokenizer everywhere (now resolved via `bharat/tokenizer/`)
- No loss masking in SFT (now resolved in `bharat/posttraining/`)
- Batch-level prompt length in DPO (now resolved)
- No checkpoint metadata (now resolved)
- Wildcard CORS in API (now configurable)
- Zero tests (now at 177+ CPU tests in CI)

## Modern Components (Milestone 2.1)

The following standalone model components are implemented in `bharat/models/`:

| Component | File | Description |
|-----------|------|-------------|
| `BharatModelConfig` | `config.py` | Frozen dataclass with derived properties (`head_dim`, `num_key_value_groups`) and validation |
| `RMSNorm` | `normalization.py` | Root-mean-square layer normalization with learnable scale; float32 variance |
| `RotaryEmbedding` | `rotary.py` | Interleaved even/odd rotation; cached inv_freq; explicit position IDs; cache extension |
| `apply_rotary_pos_emb` | `rotary.py` | Pure helper to apply RoPE to Q and K tensors |
| `SwiGLU` | `mlp.py` | SiLU-gated MLP with independent gate/up/down projections |
| `GroupedQueryAttention` | `attention.py` | GQA with separate Q/K/V projections; SDPA; RoPE; K/V repeat-interleave |

The **full Bharat decoder model** is not yet implemented. KV caching is not yet implemented.
Model-size configurations are not yet validated. When a CUDA GPU with FlashAttention is available,
PyTorch SDPA selects it automatically. No benchmark or quality claims have been established.

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     bharat/ package                          │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ tokenizer│  │  models  │  │ training │  │ data     │   │
│  │          │  │          │  │          │  │          │   │
│  │ • base   │  │ • config │  │ • ckpt   │  │ • sources│   │
│  │ • loader │  │ • rotary │  │   mgmt   │  │ • dedup  │   │
│  │ • train  │  │ • norm   │  │          │  │ • PII    │   │
│  │ • eval   │  │ • mlp    │  │          │  │ • filter │   │
│  │ • meta   │  │ • attn   │  │          │  │ • shards │   │
│  └──────────┘  │ • bharat │  └──────────┘  └──────────┘   │
│                └──────────┘                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │posttrain │  │  eval    │  │ serving  │  │ safety   │   │
│  │          │  │          │  │          │  │          │   │
│  │ • SFT    │  │ • runner │  │ • api    │  │ • filter │   │
│  │ • DPO    │  │ • bench  │  │ • auth   │  │ • audit  │   │
│  │ • loss   │  │ • report │  │ • rate   │  │          │   │
│  └──────────┘  └──────────┘  │ • export │  └──────────┘   │
│                              └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
            ▲                            ▲
            │                            │
  ┌─────────┴─────────┐      ┌───────────┴──────────┐
  │   configs/        │      │   data/shards/       │
  │   bharat-*.yaml   │      │   train.bin, val.bin │
  └───────────────────┘      └──────────────────────┘
```

## Tokenizer Flow

```
User-facing API
      │
      ▼
bharat/tokenizer/load_tokenizer()
      │
      ├── "gpt2" → GPT2TokenizerFast → _GPT2Wrapper
      ├── path/to/tokenizer.json → HFTokenizers → _SentencePieceWrapper
      ├── path/to/model → AutoTokenizer → _HFWrapper
      └── "bert-base-uncased" → AutoTokenizer → _HFWrapper
              │
              ▼
      bharat/tokenizer/evaluate.py
      compression_ratio, fertility, language_wise_fertility
```

## Data Flow

```
Data Sources
      │
      ▼
bharat/data/registry.py ── licence validation
      │
      ▼
bharat/data/: language_id, normalization, dedup, PII, quality
      │
      ▼
bharat/data/sharding.py ── uint16/uint32 auto-detect
      │
      ▼
data/shards/{train,val}.bin + manifest.json
```

## Training Flow

```
configs/bharat-350m.yaml
      │
      ▼
bharat/models/bharat_model.py ← RoPE, RMSNorm, SwiGLU, GQA
      │
      ├── bharat/training/ ── checkpointing, resume
      │
      ├── bharat/posttraining/sft.py ── assistant-only loss mask
      │
      └── bharat/posttraining/dpo.py ── per-sample mask
               │
               ▼
      bharat/evaluation/ ── BharatBench
               │
               ▼
      bharat/serving/api.py ── streaming, auth, metrics
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for migration details.
