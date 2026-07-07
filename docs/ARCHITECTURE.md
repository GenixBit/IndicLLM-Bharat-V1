# Bharat AI Architecture

## Legacy (GPT-2)

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

## Modern Components (Milestone 2.1)

The following standalone model components are implemented in `bharat/models/`:

| Component | File | Description |
|-----------|------|-------------|
| `BharatModelConfig` | `config.py` | Frozen dataclass with derived properties (`head_dim`, `num_key_value_groups`) and validation |
| `RMSNorm` | `normalization.py` | Root-mean-square layer normalization with learnable scale; float32 variance |
| `RotaryEmbedding` | `rotary.py` | Interleaved even/odd rotation; cached inv_freq; float32 computation; explicit position IDs; cache extension |
| `apply_rotary_pos_emb` | `rotary.py` | Pure helper to apply RoPE to Q and K tensors |
| `SwiGLU` | `mlp.py` | SiLU-gated MLP with independent gate/up/down projections; dropout on final output |
| `GroupedQueryAttention` | `attention.py` | GQA with separate Q/K/V projections; SDPA; RoPE; K/V repeat-interleave; KV cache support; combined causal + padding mask |

## Full Bharat Decoder Model (Milestone 2.2)

The complete decoder-only Bharat language model is assembled in `bharat/models/`:

### Architecture

```
input_ids → embed_tokens
    → BharatDecoderLayer × num_hidden_layers
        → RMSNorm → GroupedQueryAttention → residual+
        → RMSNorm → SwiGLU → residual+
    → final RMSNorm
    → lm_head (tied weight optional)
```

### Residual and normalisation order

Pre-normalisation (norm-then-attention/MLP, residual addition after):

```
residual → RMSNorm → attention → dropout → + residual
residual → RMSNorm → SwiGLU    → dropout → + residual
```

### Cache tensor shapes

Each layer's KV cache stores unexpanded key/value tensors:

```
key:   (batch_size, num_key_value_heads, cached_length, head_dim)
value: (batch_size, num_key_value_heads, cached_length, head_dim)
```

Only `num_key_value_heads` heads (not the expanded query-head count) are stored. MHA, GQA, and MQA modes all use the same storage convention.

### Cached causal-mask design

When a KV cache is present:

- The causal mask is built explicitly as `(1, 1, query_length, key_length)` using `torch.triu` with `diagonal=1`.
- Padding mask (if any) is added element-wise.
- `is_causal=False` when a mask is supplied, ensuring `scaled_dot_product_attention` does not add its own causal mask.
- For single-token decode, query_length=1 and key_length=past_len+1; only the single query row is computed.

### Position-ID handling

- Without cache: auto-created from `past_len + arange(seq_len)`, or padding-aware cumulative sums when `attention_mask` is present and `past_len == 0`.
- With cache: explicit `position_ids` should be passed for each decode step (offset by past length).
- Rejects negative position IDs; validates length matches `seq_len`.

### Weight tying

When `config.tie_word_embeddings=True`, `lm_head.weight` is the same `Parameter` object as `model.embed_tokens.weight`. Saving and loading preserves identity.

### Loss shifting

Standard causal LM loss: `shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]`, cross-entropy with `ignore_index=-100`.

### Generation algorithms

`generate()` in `generation.py`:

1. First forward pass: full prompt with cache enabled.
2. Subsequent passes: single token with cache.
3. Supports greedy, temperature sampling, top-k, top-p.
4. EOS stopping, PAD replacement for finished sequences.
5. Batched generation with right-padded prompts.
6. Attention mask extended each step for new key positions.

### Save/load format

- `config.json`: `BharatModelConfig.to_dict()` with `model_format_version: "bharat-v1"`.
- `model.pt`: `model.state_dict()` via `torch.save`.
- Atomic temporary-file-and-rename pattern.
- Strict loading: mismatched keys raise `RuntimeError`.
- Incompatible format versions raise `ValueError`.

## Typed output dataclasses

| Class | Fields |
|-------|--------|
| `BharatModelOutput` | `last_hidden_state`, `past_key_values` |
| `BharatCausalLMOutput` | `logits`, `loss`, `past_key_values` |

## Cache validation

`validate_cache()` checks:
- layer count
- batch size
- KV head count
- head dimension
- key/value shape equality
- consistent cached length across layers
- device and dtype compatibility

`reorder_cache()` supports batch-index reordering for future beam search.

## Public API (`bharat/models/__init__.py`)

### Model components

```
BharatModelConfig
BharatDecoderLayer
BharatModel
BharatForCausalLM
BharatModelOutput
BharatCausalLMOutput
KeyValueCache
PastKeyValues
GroupedQueryAttention
RMSNorm
RotaryEmbedding
apply_rotary_pos_emb
SwiGLU
generate
past_length
reorder_cache
validate_cache
```

### Model specs and sizing

```
BharatModelSpec
ModelSpecResolver
load_model_spec
load_model_config
ParameterCount
StaticMemoryReport
KVCacheMemoryReport
calculate_parameter_count
calculate_static_memory
calculate_kv_cache_memory
```

## Config YAML files

Four validated configurations live in `configs/models/`:

| File | Nominal | Analytical params | Difference |
|------|---------|-------------------|------------|
| `configs/models/bharat-350m.yaml` | 350M | 347,393,024 | −0.74% |
| `configs/models/bharat-1b.yaml` | 1B | 999,368,704 | −0.06% |
| `configs/models/bharat-3b.yaml` | 3B | 3,009,039,360 | +0.30% |
| `configs/models/bharat-7b.yaml` | 7B | 7,040,405,504 | +0.58% |

All configurations use RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, bias-free projections, and a 64K vocabulary — see [MODEL_CONFIGURATIONS.md](MODEL_CONFIGURATIONS.md) for full architecture tables.

## Legacy GPT-2

The legacy GPT-2 model (`train/pretrain.py`, `GPT`, `GPTConfig`) remains fully supported. No legacy code has been deleted or modified.

## Remaining limitations

- No Bharat model has been pretrained or benchmarked.
- No performance or quality claims are established.
- FlashAttention is not explicitly integrated (PyTorch SDPA selects it automatically when CUDA is available).
- No distributed training, quantization, or serving integration for the Bharat model.

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
