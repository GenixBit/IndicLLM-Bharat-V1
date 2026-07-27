# Milestone 6.2 — Deterministic BPE Tokenizer Training

## Status: Draft

## Objective

Build a standalone, deterministic byte-level BPE training harness that produces a
reproducible tokenizer vocabulary, merge list, and tokenizer hash from a given
JSONL corpus. This component is the second building block (after the corpus
sampler) for the Bharat-350M tokenizer pipeline.

## Scope

- Byte-level BPE training (`train_bpe`)
- Complete 256-byte alphabet as initial vocabulary
- Configurable special tokens with stable IDs
- Deterministic vocabulary, merges, and tokenizer hash
- Save/load round-trip for trained tokenizer
- CLI (`scripts/train_tokenizer.py`)
- CPU-only, fully offline, no external tokenizer library
- All tests use synthetic or sampler-generated fixtures

## Out of Scope

- Production 64K vocabulary training (separate pipeline)
- Tokenization/encoding API (Milestone 6.3)
- Evaluation (Milestone 4)
- GPU support

## Files

| File | Purpose |
|------|---------|
| `bharat/tokenizer/bpe.py` | Core BPE training logic |
| `scripts/train_tokenizer.py` | CLI wrapper |
| `tests/tokenizer/test_bpe.py` | Unit tests (vocabulary, training, determinism, serialization) |
| `tests/scripts/test_train_tokenizer.py` | CLI tests |

## Determinism Guarantee

Given identical inputs (corpus file, vocab_size, special tokens), repeated
training runs produce identical:
- Vocabulary dict (keys and IDs)
- Merge list (order, left/right IDs, token IDs)
- `tokenizer_hash` (SHA-256 of corpus, special tokens, and config)

A 3-second delayed execution test confirms temporal independence.

## Vocabulary Layout

| IDs | Type |
|-----|------|
| 0–3 (configurable) | Special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`) |
| 4–259 | 256 byte tokens (`<byte_00>` through `<byte_ff>`) |
| 260+ | Learned BPE merges |

## Verification

```bash
# Unit tests
pytest tests/tokenizer/test_bpe.py -v

# CLI tests
pytest tests/scripts/test_train_tokenizer.py -v

# Full verification
ruff format --check .
ruff check .
mypy bharat/
```
