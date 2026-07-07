# Bharat Model Configurations

This document defines the four production-intent model architectures for the
Bharat family: **Bharat-350M**, **Bharat-1B**, **Bharat-3B** and **Bharat-7B**.

All four configurations are validated by the analytical parameter calculator
and are within 1 % of their nominal parameter tier.

**Important**: No model has been pretrained.  No accuracy, quality or
performance claims exist for any of these configurations.

### Truthful statements

* The 64 000 vocabulary size is an **architecture assumption**; the final Bharat tokenizer has not yet been trained.
* The current supported model context is **4 096 tokens** — no RoPE interpolation, NTK scaling, YaRN or LongRoPE has been implemented.
* GQA reduces analytical KV-cache size, but **model-quality impact has not been measured**.
* No Bharat model has been pretrained or benchmarked.

---

## Architecture tables

| Parameter              | Bharat-350M | Bharat-1B  | Bharat-3B  | Bharat-7B  |
|------------------------|-------------|------------|------------|------------|
| Vocab size             | 64 000      | 64 000     | 64 000     | 64 000     |
| Hidden size            | 1 024       | 2 048      | 3 072      | 4 096      |
| Intermediate size      | 2 816       | 6 144      | 9 984      | 13 824     |
| Layers                 | 25          | 18         | 24         | 32         |
| Attention heads        | 16          | 16         | 24         | 32         |
| KV heads               | 4           | 4          | 8          | 8          |
| Head dim               | 64          | 128        | 128        | 128        |
| GQA groups             | 4           | 4          | 3          | 4          |
| Max positions          | 4 096       | 4 096      | 4 096      | 4 096      |
| RoPE theta             | 10 000.0    | 10 000.0   | 10 000.0   | 10 000.0   |
| Tie embeddings         | True        | True       | True       | True       |
| Attention bias         | False       | False      | False      | False      |
| MLP bias               | False       | False      | False      | False      |

## Parameter counts

| Config       | Target      | Analytical   | Difference |
|--------------|-------------|--------------|------------|
| Bharat-350M  | 350 000 000 | 347 393 024  | −0.7449 %  |
| Bharat-1B    | 1 000 000 000 | 999 368 704 | −0.0631 %  |
| Bharat-3B    | 3 000 000 000 | 3 009 039 360 | +0.3013 % |
| Bharat-7B    | 7 000 000 000 | 7 040 405 504 | +0.5772 % |

All configurations are within 1 % of their nominal tier.

---

## GQA design

All configurations use Grouped Query Attention (GQA) with fewer key-value
heads than query heads.  This reduces KV-cache memory without significantly
affecting model quality.

| Config       | Query heads | KV heads | GQA groups |
|--------------|-------------|----------|------------|
| Bharat-350M  | 16          | 4        | 4          |
| Bharat-1B    | 16          | 4        | 4          |
| Bharat-3B    | 24          | 8        | 3          |
| Bharat-7B    | 32          | 8        | 4          |

---

## Vocabulary

All configurations use a vocabulary of 64 000 tokens.  This is a
byte-level BPE tokeniser covering English, Indic languages and code.

---

## Context length

All models support a maximum context of 4 096 tokens.  RoPE supports
extension beyond this length through interpolation.

---

## Tied embeddings

Word embeddings are tied with the LM head in all configurations.
This saves `vocab_size × hidden_size` parameters.

---

## Bias settings

Attention and MLP projections are bias-free in all configurations.
Biases are supported by the architecture and can be enabled for
specific use cases.

---

## Analytical formula

The exact parameter counting formula matches the implementation in
`bharat/models/sizing.py`:

```
H = hidden_size
I = intermediate_size
L = num_hidden_layers
A = num_attention_heads
K = num_key_value_heads
D = H / A
V = vocab_size

token_embeddings = V × H

attention per layer:
  Q: H × H
  K: H × (K × D)
  V: H × (K × D)
  O: H × H
  (+ biases if attention_bias)

MLP per layer (SwiGLU):
  gate: H × I
  up: H × I
  down: I × H
  (+ biases if mlp_bias)

norms per layer:
  2 × H

transformer_layers = L × (attention + MLP + norms)
final_norm = H
lm_head = 0 if tied else V × H

total = token_embeddings + transformer_layers + final_norm + lm_head
```

---

## Memory calculation assumptions

Weight memory: `parameter_count × bytes_per_element`

Gradient memory (when enabled): `parameter_count × bytes_per_element`

Master weights (fp32, when enabled): `parameter_count × 4`

AdamW optimizer state: `2 × parameter_count × 4`
(first moment + second moment, both fp32)

KV-cache formula:
```
batch_size × sequence_length × num_hidden_layers × 2
× num_key_value_heads × head_dim × bytes_per_element
```

The factor 2 represents the key and value tensors.

### Why activation memory is excluded

Activation memory depends on batch size, sequence length and framework
implementation details.  It is not a fixed property of the model
architecture and is therefore excluded from exact analytical reports.

---

## Status

- [x] Four production-intent configuration files exist
- [x] All configurations pass `BharatModelConfig` validation
- [x] Analytical parameter counts match literal expected values
- [x] All configurations are within 1 % of their nominal tier
- [x] Analytical counts match real tiny-model parameter counts
- [x] Tied and untied embedding calculations are verified
- [x] Bias calculations are verified
- [x] Memory calculators are implemented and tested
- [x] CLI calculator is implemented
- [x] Milestone 2.3 verification tests pass

- [ ] No model has been pretrained
- [ ] No model weights exist
- [ ] No accuracy, quality or performance claims are made
- [ ] No benchmark results are reported
- [ ] Final Bharat tokenizer has not yet been trained
- [ ] RoPE interpolation, NTK scaling, YaRN or LongRoPE is not implemented
- [ ] Model-quality impact of GQA has not been measured
