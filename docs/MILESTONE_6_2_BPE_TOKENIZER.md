# PR C — Deterministic BPE Tokenizer Training Harness

## Status: In review

## Objective

Build a standalone, deterministic byte-level BPE training harness that produces a
reproducible tokenizer vocabulary, merge list, and tokenizer hash from a given
JSONL corpus. This is PR C, a component of the broader Milestone 6.1 pipeline
(corpus sampling → BPE training → final tokenizer assembly).

## Token-ID Layout

| IDs | Type |
|-----|------|
| 0–3 | Special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`) |
| 4–259 | 256 byte tokens (`<byte_00>` through `<byte_ff>`) |
| 260+ | Learned BPE merge tokens |

### Special-Token Contract

- `<pad>` = 0, `<unk>` = 1, `<bos>` = 2, `<eos>` = 3 (aligned with Milestone 6.1 plan)
- IDs are non-negative and unique
- Ordinary encoding of literal text like `"<pad>"` does NOT emit special IDs unless `allow_special=True`
- `<unk>` is retained for compatibility; with complete byte coverage, valid input text never produces it

### Byte Mapping

- `byte_value_to_id`: maps raw byte value (0–255) → token ID (4–259)
- `id_to_bytes`: maps token ID → canonical byte payload
- Merge tokens use `id_to_bytes[new_id] = id_to_bytes[left] + id_to_bytes[right]`
- All tokens store their canonical byte content, not string-concatenated names

### Merge Representation

- Each merge is stored as `(left_id, right_id, token_id, rank)`
- Merge rank is the training step index (0-based)
- Tie-breaking: when multiple pairs share the same maximum frequency, the pair with the smallest `(left_id, right_id)` is chosen

## JSONL Input Contract

- Read each line as a JSON object
- Extract the configured `text` field (default: `"text"`)
- Records are treated as separate sequences; pairs never cross record boundaries
- JSON syntax (field names, punctuation, newlines) is never included as training text
- Rejected: malformed UTF-8, malformed JSON, non-object records, missing text, non-string text, lone surrogates
- Empty records are silently skipped

## Normalization

- Normalization policy: NFC
- Applied prior to encoding; training text is used as-is from the JSONL field

## Encode/Decode

### encode(text, *, allow_special=False)
1. Reject lone surrogates
2. Optionally parse special tokens if `allow_special=True`
3. Map each character's UTF-8 bytes to byte-token IDs via `byte_value_to_id`
4. Apply learned merges in rank order
5. Return the final ID sequence

### decode(ids)
1. Reject unknown IDs
2. Concatenate canonical token bytes from `id_to_bytes`
3. Decode valid UTF-8 strictly
4. Special tokens expand to empty string (their IDs are skipped)

## Artifact Schema

```json
{
  "schema_version": "bpe-v1",
  "normalization": "nfc",
  "special_tokens": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
  "reserved_tokens": {},
  "byte_value_to_id": {"0": 4, "1": 5, ..., "255": 259},
  "id_to_bytes": {"4": "00", "5": "01", ..., "259": "ff"},
  "vocab": {"<pad>": 0, ..., "<byte_00>": 4, ..., "<merge_0>": 260, ...},
  "merges": [[97, 98, 260, 0], ...],
  "tokenizer_hash": "sha256hex"
}
```

## Tokenizer Hash

SHA-256 of the canonical JSON payload covering:
- schema_version, normalization
- special_tokens, reserved_tokens
- byte_value_to_id, id_to_bytes
- vocab, merges

Verified on load: if stored hash does not match recomputed hash, loading fails.

## Save/Load Integrity

- Save: write to temp file in same directory, verify by reloading, atomic rename
- Overwrite prohibited by default (`FileExistsError`); use `overwrite=True`
- Load: hash verification against stored hash; unsupported schema versions rejected

## Determinism Evidence

Given identical inputs (corpus, vocab_size, special_tokens, seed-independent):
- Identical vocabulary
- Identical merge order and IDs
- Identical tokenizer hash
- Byte-identical serialized artifact
- All verified with 3-second delayed execution test

## Limitations

- Production 64K vocabulary training has NOT been performed and is NOT part of this PR
- Multi-GPU or distributed training is not supported
- Tokenization API is standalone; integration with `BharatTokenizer` is deferred

## Files

| File | Purpose |
|------|---------|
| `bharat/tokenizer/bpe.py` | Core BPE training, encode/decode, serialization |
| `scripts/train_tokenizer.py` | CLI wrapper |
| `tests/tokenizer/test_bpe.py` | 51 unit tests |
| `tests/scripts/test_train_tokenizer.py` | 10 CLI tests |
| `docs/MILESTONE_6_2_BPE_TOKENIZER.md` | This document |

## Verification

```bash
pytest tests/tokenizer/test_bpe.py -q
pytest tests/scripts/test_train_tokenizer.py -q
ruff format --check .
ruff check .
mypy bharat/
```
