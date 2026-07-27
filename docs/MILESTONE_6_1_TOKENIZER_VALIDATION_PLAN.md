# Milestone 6.1 — 64K BPE Tokenizer Validation Plan

**Status:** Planning (not started)
**Date:** 2026-07-27
**Branch:** `docs/milestone-6-1-tokenizer-validation-plan`
**PR sequence:** A (see [Phased PR Plan](#24-phased-pr-plan))

---

## 1. Current-State Audit

### 1.1 Working Tokenizer Abstractions

The repository provides a mature tokenizer abstraction layer in `bharat/tokenizer/`:

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `base.py` | 65 | `BharatTokenizer` abstract base class | ✅ Working |
| `loader.py` | 430 | 4 wrapper classes + `load_tokenizer()` dispatch | ✅ Working |
| `train.py` | 112 | `train_bpe_tokenizer()`, `train_sentencepiece_tokenizer()` | ✅ Working |
| `evaluate.py` | 95 | compression ratio, fertility, per-language metrics | ✅ Working |
| `metadata.py` | 78 | `TokenizerMetadata`, `tokenizer_hash()`, compatibility validation | ✅ Working |

### 1.2 Existing Wrapper Classes

| Class | Underlying Library | `tokenizer_type` | Used For |
|-------|-------------------|-------------------|----------|
| `_GPT2Wrapper` | `transformers.GPT2TokenizerFast` | `"gpt2"` | Default fallback, legacy GPT-2 |
| `_SentencePieceNativeWrapper` | `sentencepiece.SentencePieceProcessor` | `"sentencepiece"` | Native `.model` files |
| `_SentencePieceHFWrapper` | `tokenizers.Tokenizer` with SP/BPE model | `"sentencepiece"` | HF `tokenizer.json` files |
| `_HFWrapper` | `transformers.PreTrainedTokenizerFast` | `"hf"` | Generic HF tokenizers |

### 1.3 Tokenizer Training Functions

**`train_bpe_tokenizer()`** (`bharat/tokenizer/train.py:9`):
- Uses HuggingFace `tokenizers` library
- Normalizer: `NFC`
- Pre-tokenizer: `ByteLevel(add_prefix_space=False)`
- Post-processor: `ByteLevel(trim_offsets=False)`
- Model: `BPE`
- Default special tokens: `["<|endoftext|>", "<|pad|>"]`
- Returns wrapped as `_SentencePieceHFWrapper` (tokenizer_type `"sentencepiece"` — a misnomer for BPE)

**`train_sentencepiece_tokenizer()`** (`bharat/tokenizer/train.py:51`):
- Uses Google `sentencepiece` library
- Supports `model_type="bpe"` or `model_type="unigram"`
- Default IDs: pad=0, unk=1, bos=2, eos=3
- Returns wrapped as `_SentencePieceNativeWrapper`

### 1.4 Evaluation Functions

All in `bharat/tokenizer/evaluate.py`:

| Function | Metric | Accepts |
|----------|--------|---------|
| `compression_ratio()` | chars / tokens | `texts: list[str]` |
| `fertility()` | tokens / words | `texts: list[str]` |
| `top_k_rare_tokens()` | least frequent tokens | `texts: list[str]`, `k: int` |
| `top_k_common_tokens()` | most frequent tokens | `texts: list[str]`, `k: int` |
| `language_wise_fertility()` | fertility per language | `texts_by_lang: dict[str, list[str]]` |
| `code_efficiency()` | compression ratio for code | `code_snippets: list[str]` |
| `all_metrics()` | all metrics combined | `texts: list[str]` |

**Missing metrics** (not yet implemented but required):
- Tokens per character (inverse of compression ratio)
- Tokens per byte
- Unknown-token rate
- Byte-fallback rate
- Round-trip fidelity
- Script fragmentation
- Word fragmentation
- Numeric fragmentation
- Code-token fragmentation
- Whitespace preservation
- Special-token correctness

### 1.5 Metadata and Hashing

`bharat/tokenizer/metadata.py` provides:
- `TokenizerMetadata` — frozen dataclass with type, vocab_size, hash, special tokens, config, git_sha, data_version, seed
- `tokenizer_hash(tokenizer)` — delegates to `.fingerprint()` on `BharatTokenizer` → SHA-256 hex
- `metadata_from_tokenizer(tokenizer)` — builds metadata from live tokenizer
- `validate_tokenizer_compatibility(ckpt_meta, tokenizer)` — raises `ValueError` on hash mismatch

### 1.6 Checkpoint Compatibility

`bharat/training/checkpointing.py`:
- `save_checkpoint()` stores `tokenizer_type`, `tokenizer_hash`, `vocab_size`
- `load_checkpoint()` validates tokenizer compatibility on load
- `validate_checkpoint()` compares hash and vocab size

### 1.7 Missing Production 64K Tokenizer Requirements

| Requirement | Status |
|-------------|--------|
| 64K byte-level BPE training config | ❌ Not defined |
| Fixed special-token IDs | ❌ Not defined |
| Normalization policy with Indic tests | ❌ Not defined |
| Deterministic sampling from approved data | ❌ Not implemented |
| Per-language evaluation metrics | ❌ Not implemented (API exists, no fixtures) |
| Baseline comparison (GPT-2) | ❌ Not implemented |
| Acceptance thresholds | ❌ Not defined |
| Tokenizer-specific wrapper (`type="bpe"`) | ❌ Not implemented (uses sentencepiece type) |
| Tokenizer corpus digest | ❌ Not implemented |
| Tokenizer registry entry | ❌ Not implemented |

---

## 2. Algorithm Comparison

### 2.1 Candidates Evaluated

| Algorithm | Library | Ecosystem | Indic Support | Unknown Tokens | Deterministic |
|-----------|---------|-----------|---------------|----------------|---------------|
| **Byte-level BPE** | `tokenizers` | Llama 2/3, Mistral, CodeLlama, Gemma 2 | Full (bytes cover all Unicode) | None | Yes |
| SentencePiece BPE | `sentencepiece` | T5, Gemma 1, Llama 1 | Good with `byte_fallback` | Possible without fallback | Yes |
| SentencePiece Unigram | `sentencepiece` | T5, Gemma | Good with `byte_fallback` | Possible without fallback | Subword-sampling mode non-deterministic |

### 2.2 Recommended Algorithm

**Byte-level BPE** via the `tokenizers` library.

### 2.3 Rationale

1. **Zero unknown tokens**: Byte-level BPE operates on UTF-8 bytes. Every possible Unicode string is tokenizable. This is critical for Indic languages where a vocabulary may not cover all Unicode code points.

2. **Inference ecosystem compatibility**: All major Llama-family models (Llama 2, Llama 3, Mistral, CodeLlama, Qwen 2) use byte-level BPE. Bharat-350M uses the same architecture family (RoPE, RMSNorm, SwiGLU, GQA). Using the same tokenizer family maximizes ecosystem compatibility for inference servers, quantization tools, and evaluation frameworks.

3. **Whitespace preservation**: Byte-level BPE does not normalize whitespace. This is important for code (Python indentation) and Indic languages where whitespace rules differ from English.

4. **No additional runtime dependency**: `tokenizers>=0.19.0` is already a project dependency. The `sentencepiece` library is also a dependency but adds complexity (two tokenizer backends).

5. **Deterministic**: Training is fully deterministic given fixed seed, data order, and parameters.

6. **Existing training harness**: `train_bpe_tokenizer()` in `bharat/tokenizer/train.py` provides a working foundation that needs configuration hardening rather than rewriting.

### 2.4 Unigram Not Selected

SentencePiece Unigram can be effective but:
- Subword-regularization mode is non-deterministic (sampling from multiple segmentations)
- Inference ecosystem support is weaker than byte-level BPE
- Whitespace normalization (SentencePiece replaces spaces with `_`) complicates code and Indic text
- Not worth the complexity for Bharat-350M's first tokenizer

### 2.5 SentencePiece BPE Not Selected

- Adds a second tokenizer training backend (already have `tokenizers`)
- Whitespace normalization interferes with code tokenization
- `byte_fallback` is required for full Unicode coverage (adds complexity)
- The `tokenizers` library handles byte-level BPE natively and is better integrated

---

## 3. 64K Vocabulary Composition

### 3.1 Total Count

**64,000 tokens** — matching `configs/models/bharat-350m.yaml`

### 3.2 Composition (Training Order)

| Range | Count | Type | Description |
|-------|-------|------|-------------|
| 0–2 | 3 | Special tokens | `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>` |
| 3 | 1 | Reserved | Future special token |
| 4–7 | 4 | Reserved | Chat template markers (system, user, assistant, tool) |
| 8–10 | 3 | Reserved | Future use |
| 11–266 | 256 | Byte tokens | Raw UTF-8 bytes 0x00–0xFF |
| 267–63999 | 63,733 | Learned tokens | BPE merges learned from training corpus |

**Total: 3 + 1 + 4 + 3 + 256 + 63,733 = 64,000**

### 3.3 Rationale for Reserving IDs

Reserved IDs ensure that special tokens can be added later without changing the embedding table size. The 8 reserved slots (IDs 3–10) are a reasonable buffer — enough for chat template tokens and control tokens, small enough to not waste significant vocabulary capacity.

### 3.4 Byte Token Range

IDs 11–266 represent raw bytes 0x00–0xFF. Byte-level BPE inherently covers all bytes; the specific ID assignment depends on training order. These IDs are stable once training completes.

### 3.5 Vocabulary File

The trained tokenizer must produce:
- A `tokenizer.json` file (HuggingFace `tokenizers` format)
- A `vocab.json` or equivalent vocabulary listing
- Config JSON with special-token ID mappings

### 3.6 Model Config Alignment

The model configuration (`vocab_size: 64000`) must exactly match the tokenizer's output vocabulary size. Verified by `validate_tokenizer_compatibility()`.

---

## 4. Special-Token Contract

### 4.1 Required Tokens and Stable IDs

| Token | ID | Purpose | Immutable |
|-------|----|---------|-----------|
| `<\|pad\|>` | 0 | Padding | Yes |
| `<\|bos\|>` | 1 | Beginning of sequence | Yes |
| `<\|eos\|>` | 2 | End of sequence | Yes |
| *Reserved* | 3 | Future use | N/A |
| *Reserved* | 4–7 | Chat template markers | N/A |
| *Reserved* | 8–10 | Future use | N/A |

### 4.2 No UNK Token

Byte-level BPE has no unknown tokens — every byte sequence is valid. An UNK token is not needed and will not be added.

### 4.3 Special-Token Handling in Training

- Special tokens are added to the tokenizer before training (`special_tokens` parameter of `BpeTrainer`)
- They receive the lowest IDs (0, 1, 2) by virtue of being first in the special_tokens list
- They are not learned — they are excluded from the BPE merge learning

### 4.4 Adding Special Tokens After Training

If additional special tokens (e.g., `<|system|>`, `<|user|>`, `<|assistant|>`) are needed later, they must be added via `add_special_tokens` with a corresponding increase in `vocab_size` in the model config. The production wrapper must support this operation.

---

## 5. Normalization Policy

### 5.1 Normalization Algorithm

**NFC (Normalization Form C)** — Unicode canonical composition.

### 5.2 Decision Rationale

- NFC is the standard normalization form for web text
- Most Indic scripts are already in NFC in their standard representation
- NFC is safe for all Indic scripts because Unicode defines no compatibility decompositions for most Indic characters (Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia)
- NFC is the current default in `train_bpe_tokenizer()` (`normalizers.NFC()`)
- NFC ensures that "é" (NFC: U+00E9) and "é" (NFD: U+0065 U+0301) produce the same token sequence

### 5.3 Confirmed Safe Characters

Each of the following categories must pass round-trip fidelity tests (encode → decode == original):

| Category | Example | Risk |
|----------|---------|------|
| ASCII | `a`, `0`, `+` | None |
| Devanagari | `नमस्ते` | None |
| Bengali | `বাংলা` | None |
| Gurmukhi | `ਸਤ ਸ੍ਰੀ ਅਕਾਲ` | None |
| Gujarati | `નમસ્તે` | None |
| Tamil | `வணக்கம்` | None |
| Telugu | `నమస్కారము` | None |
| Kannada | `ನಮಸ್ಕಾರ` | None |
| Malayalam | `നമസ്കാരം` | None |
| Odia | `ନମସ୍କାର` | None |
| Assamese | `নমস্কাৰ` | None |
| Zero-width joiner | ZWJ sequences (e.g., क्ष in Devanagari via ZWJ) | Must survive NFC unchanged |
| Zero-width non-joiner | ZWNJ in Persian/Arabic/Indic | Must survive NFC unchanged |
| Combining marks | Multiple diacritics on one base | May be reordered; verify identity |
| Emoji | `😀`, `👨‍👩‍👧‍👦` (ZWJ sequences) | Must round-trip |
| Malformed UTF-8 | Invalid byte sequences | Byte-level BPE handles; verify crash-free |

### 5.4 Prohibited Normalizations

- **NFKC** — rejected. NFKC can destructively normalize characters (e.g., ⁴ → 4, ½ → 1/2, ℕ → N). This is information-destructive and unacceptable for a language model tokenizer.
- **NFD** — not selected. NFD changes byte representation of text (decomposes characters). While reversible through NFC, it's unnecessary complexity.
- **Case folding** — not selected. Case folding loses information (e.g., "Apple" vs "apple").
- **Whitespace normalization** — not selected. Byte-level BPE preserves whitespace; any whitespace normalization would break code and multilingual text.

### 5.5 Behavior for Edge Cases

| Case | Behavior |
|------|----------|
| Tabs | Preserved as `\t` (byte 0x09) |
| Repeated spaces | Preserved |
| Line endings | `\n` preserved (byte 0x0A); `\r\n` preserved as two bytes |
| BOM | U+FEFF preserved (byte sequence EF BB BF) |
| Null byte | Preserved (byte 0x00), rarely occurs in training data |
| Very long lines | No truncation; tokenizer processes all bytes |
| Empty string | Returns empty list (no special tokens added) |
| String with only special tokens | Returns special token IDs if `add_special_tokens=True` |

### 5.6 Tests Required

Before production training, unit tests must verify:
- NFC round-trip for each Indic script listed above
- NFC identity for characters that are identical in NFC and NFD (most of Indic)
- Zero-width joiner/non-joiner preservation
- Emoji ZWJ sequence round-trip
- Malformed UTF-8 produces no crash (produces valid byte sequence or error)
- Empty string handling
- Null byte handling

---

## 6. Training-Data Contract

### 6.1 Approved Source Constraint

Tokenizer training data must come exclusively from:

1. **Approved local dataset releases** — built by `build-dataset-release` CLI, validated by `validate-dataset-approval`, signed by `DatasetApproval`
2. **Immutable manifests** — SHA-256 pinned, versioned
3. **Approved licenses** — commercial_use_allowed=true, model_training_allowed=true in the data registry
4. **Completed PII review** — verified by the quality pipeline (Milestone 3.2)
5. **Completed contamination review** — verified by the contamination checker (Milestone 3.3)
6. **Completed safety review** — verified review record exists
7. **Verified shard hashes** — every shard matches its recorded digest

### 6.2 Rejected Sources

| Source | Reason |
|--------|--------|
| Remote URLs | Not governed, no immutable pin |
| Unapproved registry records | Missing `ALLOW` decision |
| Unpinned revisions | Not immutable |
| Missing approvals | No `DatasetApproval` record |
| Changed shard hashes | Integrity violation |
| Unreviewed text | PII, contamination, or safety risk |
| Test or benchmark data | Data leakage |

### 6.3 Per-Source Sampling Limits

To prevent any single source from dominating the tokenizer vocabulary:

- **Maximum records per source**: 1,000,000 records
- **Maximum bytes per source**: 500 MB (raw UTF-8)
- **Maximum records per language**: 500,000 records
- **Maximum bytes per language**: 250 MB (raw UTF-8)

These limits are provisional and will be refined during the sampling implementation (PR B).

---

## 7. Deterministic Sampling

### 7.1 Sampling Parameters

| Parameter | Value |
|-----------|-------|
| Seed | `42` |
| File ordering | Sort by file path (lexicographic, OS-independent) |
| Record ordering | Sort by (source, shard_id, record_index) |
| Duplicate handling | Remove exact byte-level duplicates within and across sources |
| Max UTF-8 bytes per record | `100,000` |
| Max records per source | `1,000,000` |
| Max records per language | `500,000` |
| Language label | From dataset manifest language field |

### 7.2 Sample Manifest

Each sampling run produces a JSON manifest containing:

```json
{
  "schema_version": 1,
  "seed": 42,
  "created_at": "2026-...",
  "sources": [
    {
      "source_id": "source_name@version",
      "approval_hash": "sha256-...",
      "release_digest": "sha256-...",
      "records_sampled": 100000,
      "bytes_sampled": 50000000,
      "language": "hi"
    }
  ],
  "corpus_digest": "sha256-...",
  "total_records": 500000,
  "total_bytes": 250000000
}
```

### 7.3 Corpus Digest

A SHA-256 hash computed over the canonical JSON of the sample manifest. This digest identifies the exact corpus used for tokenizer training. The digest must be recorded in the tokenizer metadata.

### 7.4 Reproducibility Guarantee

Repeated construction from identical approved inputs (same release versions, same seed, same parameters) must produce:
- The same record set
- The same file ordering
- The same record ordering
- The same corpus digest

---

## 8. Evaluation Fixtures and Datasets

### 8.1 Synthetic Evaluation Fixtures

For unit tests and PR-level validation, deterministic synthetic fixtures:

| Fixture | Content | Languages | Records | Purpose |
|---------|---------|-----------|---------|---------|
| `tiny_indic` | Common phrases, numbers, punctuation | hi, mr, bn, gu, pa, ta, te, kn, ml, or, as | 100 | Per-language metrics |
| `tiny_english` | Sentences, questions, paragraphs | en | 100 | Baseline comparison |
| `tiny_code` | Python, JavaScript, HTML, shell snippets | en (code) | 50 | Code efficiency |
| `tiny_mixed` | Code-mixed Indic-English (Hinglish, etc.) | hi-en, mr-en | 50 | Mixed evaluation |
| `tiny_special` | Special tokens, edge cases, empty, BOM | N/A | 20 | Correctness |
| `tiny_unicode` | Edge Unicode: ZWJ, ZWNJ, emoji, combining | N/A | 30 | Normalization |

All fixtures are committed as Python string arrays in the test suite. No data download required.

### 8.2 Evaluation Corpus

For the production tokenizer evaluation report, a small held-out sample from each approved source:
- 10,000 records per language (not used in tokenizer training)
- Sampled deterministically (seed=999, same ordering rules)
- Stored only as part of the evaluation run (not checked in)

---

## 9. Evaluation Metrics

### 9.1 Required Metrics

| Metric | Definition | Higher/Lower Better |
|--------|------------|---------------------|
| **Compression ratio** | `chars / tokens` | Higher |
| **Tokens per byte** | `tokens / bytes` | Lower |
| **Characters per token** | `chars / tokens` | Higher |
| **Fertility** | `tokens / words` | Lower |
| **Unknown-token rate** | `unknown_tokens / total_tokens * 100` | 0.0% |
| **Byte-fallback rate** | `fallback_bytes / total_chars * 100` | 0.0% (byte-level) |
| **Round-trip fidelity** | `decode(encode(text)) == text` rate | 100% |
| **Script fragmentation** | `unique_tokens_per_script / script_chars` | Lower |
| **Word fragmentation** | `tokens / unique_words` | Lower |
| **Numeric fragmentation** | `tokens_covering_numbers / number_chars` description | See below |
| **Code-token fragmentation** | `tokens / code_tokens` for code snippets | Lower |
| **Whitespace preservation** | Ratio of whitespace chars preserved in round-trip | 1.0 |
| **Special-token correctness** | Special token IDs correctly preserved in encode/decode | 100% |

### 9.2 Reporting Breakdown

Each metric must be reported separately for:

- **Per language**: hi, mr, bn, gu, pa, ta, te, kn, ml, or, as, en
- **Per script**: Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia, Latin (en)
- **Per domain**: general, code, math, news, conversation
- **Code-mixed**: Indic-English mixed text
- **Global**: Overall average

### 9.3 Metric Implementation

The evaluation framework must extend `bharat/tokenizer/evaluate.py` with:

```python
def tokens_per_byte(tokenizer, texts) -> float
def unknown_token_rate(tokenizer, texts) -> float
def byte_fallback_rate(tokenizer, texts) -> float
def round_trip_fidelity(tokenizer, texts) -> float
def script_fragmentation(tokenizer, texts, script) -> float
def word_fragmentation(tokenizer, texts) -> float
def numeric_fragmentation(tokenizer, texts) -> float
def whitespace_preservation(tokenizer, texts) -> float
def special_token_correctness(tokenizer, test_cases) -> dict[str, bool]
def comprehensive_report(tokenizer, texts_by_lang, code_snippets) -> dict
```

---

## 10. Baseline Comparison

### 10.1 Baseline Tokenizer

**GPT-2 tokenizer** (`load_tokenizer("gpt2")`) — 50,257 vocab, byte-level BPE.

This is the current default tokenizer in the repository. All training and evaluation pipelines default to it.

### 10.2 Comparison Metrics

| Metric | Indic Expectation | English Bound | Code Bound |
|--------|-------------------|---------------|------------|
| Compression ratio (Indic) | ≥ 1.5× GPT-2 | N/A | N/A |
| Compression ratio (English) | N/A | ≥ 95% of GPT-2 | N/A |
| Code compression ratio | N/A | N/A | ≥ 90% of GPT-2 |
| Unknown-token rate | 0.0% (both) | 0.0% (both) | 0.0% (both) |
| Round-trip fidelity | ≥ GPT-2 | ≥ GPT-2 | ≥ GPT-2 |

### 10.3 Acceptance Criteria for Baseline

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Indic compression improvement | ≥ 25% better than GPT-2 | Compression ratio on Indic eval set |
| English regression limit | ≤ 5% worse than GPT-2 | Compression ratio on English eval set |
| Code regression limit | ≤ 10% worse than GPT-2 | Compression ratio on code eval set |
| Round-trip correctness | 100% on all eval sets | Encode → decode identity |
| Tokenizer speed | ≥ 50% of GPT-2 throughput | Tokens/second on eval set |
| Metadata reproducibility | Identical hash for identical config | `tokenizer_hash()` |

---

## 11. Provisional Acceptance Criteria

### 11.1 Hard Gates (Must Pass Before Production Training)

| # | Criterion | Evidence |
|---|-----------|----------|
| 1 | Exact round-trip for all supported scripts | Unit test with synthetic fixtures |
| 2 | Zero accidental special-token insertion | encode("text") returns no special token IDs inside text |
| 3 | Deterministic training output | Two identically-configured training runs produce identical tokenizer hash |
| 4 | Stable tokenizer hash | `tokenizer_hash()` returns same 64-char SHA-256 for same config |
| 5 | Stable special-token IDs | VOC 0, 1, 2 are always `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>` |
| 6 | Unknown-token rate = 0.0% | Byte-level BPE encodes all bytes |
| 7 | Tokenizer vocabulary matches model config | `vocab_size == 64000` |
| 8 | Training data from approved sources only | Immutable release digest in sample manifest |
| 9 | No PII/contamination in training data | From governed data pipeline |

### 11.2 Soft Gates (Provisional, Must Be Confirmed by PR F)

| # | Criterion | Target | Notes |
|---|-----------|--------|-------|
| 10 | Indic compression ratio | ≥ 1.5× GPT-2 baseline | Measured per-language |
| 11 | English compression ratio | ≥ 95% of GPT-2 | Measured on English eval set |
| 12 | Code compression ratio | ≥ 90% of GPT-2 | Measured on code eval set |
| 13 | Per-language fertility | ≤ 2.0 for all supported languages | Lower is better |
| 14 | Round-trip fidelity | 100% | On all eval sets |
| 15 | Tokenizer throughput | ≥ 50% of GPT-2 throughput | Tokens/second |

### 11.3 Threshold Review

Gates 10–15 are marked **provisional**. They will be adjusted in a later evidence PR (PR F) once baseline measurements confirm or refute the targets.

---

## 12. Model Compatibility Checks

### 12.1 Required Checks

| Check | Location | Failure Mode |
|-------|----------|--------------|
| Tokenizer vocab == model vocab_size | Configuration validation | Reject config mismatch |
| Embedding table size matches vocab | `bharat/models/config.py` | Dimension mismatch error |
| LM head size matches vocab | `bharat/models/model.py` | Dimension mismatch error |
| Tied-weight behavior correct | `tie_word_embeddings=true` in config | Verify weight sharing |
| Checkpoint metadata contains tokenizer type, hash, vocab | `make_checkpoint_data()` | Metadata stored at save |
| Resume rejects different tokenizer | `validate_checkpoint()` | ValueError on load |
| Inference rejects mismatched tokenizer | `LocalInferenceConfig` | Validation error |
| Export manifest records tokenizer identity | `run_export_plan.py` | Metadata in export output |

### 12.2 Current State

| Check | Implemented? | Location |
|-------|-------------|----------|
| Vocab size check | ✅ | `checkpointing.py` |
| Tokenizer hash check | ✅ | `checkpointing.py` + `metadata.py` |
| Tokenizer type check | ✅ | `checkpointing.py` |
| Resume rejection | ✅ | `checkpointing.py` |
| Inference rejection | ✅ | `eval/local_inference.py` |
| Export metadata | ✅ | `run_export_plan.py` |
| Embedding/LM head dimension | ❌ Config-level | Not yet validated at init |
| Tied-weight check | ❌ Config-level | Not yet validated |

### 12.3 Model Config Updates Required

`configs/models/bharat-350m.yaml` already specifies `vocab_size: 64000`. No change needed.

A new `configs/tokenizers/bpe-64k.yaml` (or similar) should be created to hold tokenizer-specific parameters that the model config references by digest.

---

## 13. Training-Artifact Contract

### 13.1 Required Artifacts

| Artifact | Format | Size (est.) | Committed? |
|----------|--------|-------------|------------|
| Tokenizer model | `tokenizer.json` | ~10-50 MB | No (in `.gitignore`) |
| Tokenizer vocabulary | Embedded in `tokenizer.json` | (included) | No |
| Tokenizer config | JSON | ~1 KB | Yes (config only) |
| Special-token map | JSON | ~1 KB | Yes |
| Normalization config | JSON fragment | ~0.5 KB | Yes |
| Training sample manifest | JSON | ~10-100 KB | Yes |
| Training config (YAML) | YAML | ~1 KB | Yes |
| Corpus digest | 64-char hex | 64 bytes | Yes |
| Tokenizer digest | 64-char hex | 64 bytes | Yes |
| Evaluation report | JSON | ~10-50 KB | Yes |
| Approval record | `DatasetApproval` JSON | ~2 KB | Yes |

### 13.2 What Is NOT Committed

- Trained `tokenizer.json` model file (large binary)
- Training corpus (text data, already governed)
- Raw evaluation output (summarized only)
- Intermediate logging files

### 13.3 Artifact Storage

Artifacts that are too large for the git repository will be stored as governed dataset releases using the existing `build-dataset-release` and `validate-dataset-approval` pipeline.

---

## 14. Compute Budgets

### 14.1 Tiny Unit-Test Tokenizer

| Resource | Value |
|----------|-------|
| Corpus size | ~100 KB (synthetic fixtures) |
| Vocabulary | 512 |
| Memory | ~256 MB |
| CPU cores | 1 |
| Runtime | < 30 seconds |
| Temporary disk | < 10 MB |
| Output size | ~100 KB |

### 14.2 Local Smoke Tokenizer

| Resource | Value |
|----------|-------|
| Corpus size | ~10 MB (sampled from approved data) |
| Vocabulary | 8,000 |
| Memory | ~1 GB |
| CPU cores | 2 |
| Runtime | < 5 minutes |
| Temporary disk | ~100 MB |
| Output size | ~1 MB |

### 14.3 Production 64K Tokenizer

| Resource | Estimate |
|----------|----------|
| Corpus size | ~250 MB (sampled from approved data) |
| Vocabulary | 64,000 |
| Memory | ~8–16 GB |
| CPU cores | 8–16 |
| Runtime | 30–120 minutes |
| Temporary disk | ~1 GB |
| Output size | ~10–50 MB |

Production training must run outside CI (too large). Use a controlled local environment or a governed release pipeline.

---

## 15. Safety and Privacy Rules

### 15.1 Prohibited Actions

- ❌ No raw sensitive examples in reports or documentation
- ❌ No PII examples copied into test fixtures
- ❌ No network access during training or evaluation (enforced by `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- ❌ No telemetry or uploads
- ❌ No production-scale execution in CI

### 15.2 Required Protections

| Rule | Implementation |
|------|----------------|
| Synthetic test fixtures | All unit tests use hardcoded string arrays, not real data |
| Local-only corpus | Tokenizer training corpus must be pre-downloaded governed releases |
| Deterministic sampling | No randomness in source selection |
| CPU-compatible smoke evaluation | Smoke tests must run on CPU in under 5 minutes |
| No raw credentials | Environment variables only, per DATA_GOVERNANCE.md |
| No output of raw training text | Reports contain aggregated metrics, not example text |

---

## 16. Tokenizer Configuration

### 16.1 Production Configuration

```yaml
# configs/tokenizers/bpe-64k.yaml
schema_version: 1
tokenizer_name: bharat-bpe-64k
algorithm: bpe
vocab_size: 64000
normalizer: nfc
pre_tokenizer:
  type: byte_level
  add_prefix_space: false
post_processor:
  type: byte_level
  trim_offsets: false
special_tokens:
  - "<|pad|>"
  - "<|bos|>"
  - "<|eos|>"
batch_size: 10000
min_frequency: 2
seed: 42
```

### 16.2 Wrapper Type

The production tokenizer should be wrapped in a new `_BPEWrapper` class (or the existing `_SentencePieceHFWrapper` corrected) that returns:

```python
@property
def tokenizer_type(self) -> str:
    return "bpe"
```

This distinguishes it from SentencePiece-based tokenizers and matches the checkpoint metadata convention.

---

## 17. Implementation Status

### 17.1 Ready for Implementation

| Component | Depends On | Est. Effort |
|-----------|------------|-------------|
| Wrapper class for BPE (type="bpe") | Existing `_SentencePieceHFWrapper` | 1 day |
| Special-token contract in config | None | 0.5 day |
| Normalization tests | None (synthetic) | 1 day |
| Corpus sampler from local releases | Milestone 3.5 (complete) | 3 days |
| Deterministic sampling + manifest | Corpus sampler | 2 days |
| Extended evaluation metrics | Existing `evaluate.py` | 3 days |
| Baseline comparison framework | Extended metrics | 1 day |
| Acceptance threshold unit tests | Extended metrics | 1 day |
| Model compatibility additions | Existing `checkpointing.py` | 1 day |
| Tokenizer config YAML | None | 0.5 day |

### 17.2 Blocked or Deferred

| Component | Blocked By | Status |
|-----------|------------|--------|
| Production tokenizer training | Approved data release with Indic text | Blocked (data pipeline) |
| Production tokenizer evaluation report | Production tokenizer training | Deferred |
| Bharat-350M integration | Production tokenizer | Deferred |

### 17.3 Production Data Dependencies

The production tokenizer cannot be trained until:
1. At least one approved local dataset release contains sufficient Indic text (~100 MB per major language)
2. The dataset has completed all governance checks (license, PII, contamination, safety)
3. The dataset approval record exists and is valid

These dependencies are tracked in the data pipeline (Milestone 3.3–3.5).

---

## 18. Completion Criteria

Milestone 6.1 is complete when all of the following are true:

### 18.1 Documentation and Specification

- [ ] Tokenizer validation plan (this document) approved
- [ ] Tokenizer configuration YAML merged
- [ ] Special-token contract documented and IDs fixed
- [ ] Normalization policy documented with Indic tests

### 18.2 Implementation

- [ ] Deterministic tokenizer-corpus sampler implemented and tested
- [ ] Sample manifest format implemented with digest
- [ ] Extended evaluation metrics implemented (per-language, per-domain)
- [ ] Baseline comparison framework implemented
- [ ] Acceptance threshold unit tests pass
- [ ] BPE wrapper class (type="bpe") implemented
- [ ] Tokenizer metadata and hashing verified for new wrapper

### 18.3 Verification

- [ ] Synthetic fixture tests pass for all Indic scripts
- [ ] Round-trip fidelity verified for all supported scripts
- [ ] No accidental special-token insertion
- [ ] Deterministic sampling produces identical digests
- [ ] Tokenizer hash stable across identical configurations
- [ ] Model compatibility checks pass

### 18.4 Production Tokenizer

- [ ] Production 64K tokenizer trained from approved data
- [ ] Tokenizer hash recorded in checkpoint metadata
- [ ] Evaluation report generated with per-language metrics
- [ ] Acceptance thresholds confirmed or revised

---

## 19. Deferred Work

| Item | Reason | Planned Milestone |
|------|--------|-------------------|
| Chat template special tokens | Not needed until SFT training | Milestone 6.2 or 7 |
| Fill-in-the-middle (FIM) tokens | Not needed for initial pretraining | Future |
| Tool/function tokens | Not needed until tool-use fine-tuning | Future |
| Language-specific tokens | Language-identification not planned for tokenizer | Future |
| Multi-tokenizer support | One tokenizer for 350M | Future |
| Tokenizer serving endpoint | Not needed for training | Milestone 5 (production serving) |
| Subword-regularization training | Unigram not selected | Future |

---

## 20. Phased PR Plan

### PR A — Architecture and Evaluation Contract (This PR)

*Documentation only.*

Files:
- `docs/MILESTONE_6_1_TOKENIZER_VALIDATION_PLAN.md` (new)
- `docs/IMPLEMENTATION_PLAN.md` (update Milestone 6 section)

Accepts no code changes. Zero diff on production code.

### PR B — Deterministic Tokenizer-Corpus Sampler

- `scripts/sample_tokenizer_corpus.py` — CLI to sample from approved local dataset releases
- `bharat/tokenizer/sampler.py` — core sampling logic with deterministic ordering
- `tests/test_tokenizer_sampler.py` — reproducibility and digest tests
- Input: approved local dataset releases (Milestone 3.5)
- Output: sample manifest with corpus digest
- No tokenizer training yet

### PR C — Tiny Tokenizer Training Harness

- `bharat/tokenizer/train.py` — updated with production configuration support
- `bharat/tokenizer/loader.py` — add `_BPEWrapper(tokenizer_type="bpe")`
- `configs/tokenizers/bpe-64k.yaml` — production tokenizer configuration
- `tests/test_tokenizer_train.py` — reproducibility tests with synthetic data
- `tests/test_bpe_wrapper.py` — metadata, hashing, special-token tests

### PR D — Evaluation Framework

- `bharat/tokenizer/evaluate.py` — extended with all required metrics
- `tests/test_tokenizer_eval.py` — comprehensive metric tests
- `scripts/evaluate_tokenizer.py` — CLI to produce JSON evaluation report
- `tests/fixtures/tokenizer_eval.py` — per-language synthetic fixtures

### PR E — Production Training Configuration

- `configs/tokenizers/bpe-64k.yaml` — final review and approval
- Normalization policy tests
- Acceptance threshold documentation
- Compute plan verification
- No production run yet

### PR F — Production Tokenizer Evidence

*Run outside CI using approved corpus.*
- Train 64K tokenizer from sampled corpus
- Generate evaluation report with per-language metrics
- Verify tokenizer hash and metadata
- Publish only approved small artifacts (config, report, manifest, digest)
- No large model file committed

### PR G — Bharat-350M Tokenizer Integration

- Update model config if needed
- Smoke test with new tokenizer
- Overfit test with new tokenizer (small model, small data)
- Verify checkpoint metadata compatibility
- Verify export metadata compatibility

---

## 21. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| No approved dataset with Indic text by deadline | Medium | High | Prioritize Indic data sources in data pipeline |
| NFC normalization causes unexpected behavior in Indic scripts | Low | Medium | Comprehensive round-trip tests before production |
| 64K vocab insufficient for all 11 Indic languages | Low | Medium | Monitor per-language compression ratio; increase to 128K if needed |
| Byte-level BPE produces too many tokens for code | Low | Low | Code-specific evaluation; adjust sampling if needed |
| Tokenizer training too slow on available hardware | Low | Medium | Run outside CI; optimize `batch_size` and `min_frequency` |
| GPT-2 baseline comparison unfair for Indic (GPT-2 not designed for it) | Medium | Low | Acknowledge in report; add absolute quality metrics |

---

## 22. References

- `bharat/tokenizer/base.py` — `BharatTokenizer` abstract base class
- `bharat/tokenizer/loader.py` — loading dispatch with 4 wrapper classes
- `bharat/tokenizer/train.py` — `train_bpe_tokenizer()`, `train_sentencepiece_tokenizer()`
- `bharat/tokenizer/evaluate.py` — existing metrics (compression, fertility, code efficiency)
- `bharat/tokenizer/metadata.py` — `TokenizerMetadata`, `tokenizer_hash()`, compatibility
- `bharat/training/checkpointing.py` — model checkpoint compatibility
- `configs/models/bharat-350m.yaml` — model config with `vocab_size: 64000`
- `docs/IMPLEMENTATION_PLAN.md` — project implementation plan
- `docs/DATA_GOVERNANCE.md` — data governance rules
- `docs/ROADMAP.md` — project roadmap
- `docs/MODEL_CONFIGURATIONS.md` — model configuration specifications
