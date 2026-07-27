# Milestone 6.1 PR C — Tiny Deterministic BPE Training Harness

## Status: In review

This is PR C, a component of the broader Milestone 6.1 pipeline
(corpus sampling → BPE training → final tokenizer assembly).
It does NOT train the production 64K tokenizer.

## Objective

Build a standalone, deterministic byte-level BPE training harness that produces a
reproducible tokenizer vocabulary, merge list, and tokenizer hash from a given
JSONL corpus.

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

- Normalization policy: NFC (documented in artifact as `"normalization": "nfc"`)
- Applied to corpus text before UTF-8 encoding during training
- Applied to input text at the start of `encode()`
- `decode()` produces NFC output (since encoded IDs represent normalized text)
- Non-NFC valid Unicode input is normalized before encoding

### NFC round-trip behavior

- NFC input: `decode(encode(text)) == text` exactly
- Non-NFC valid Unicode: `decode(encode(text)) == unicodedata.normalize("NFC", text)`

### Special-token interaction

- Normalization happens before special-token matching in `encode()`
- This means special tokens must be in NFC form to match correctly
- Typical special tokens like `<pad>`, `<unk>`, `<bos>`, `<eos>` are ASCII and unaffected by NFC

## Encode/Decode

### encode(text, *, allow_special=False)
1. Apply NFC normalization
2. Reject lone surrogates
3. Optionally parse special tokens if `allow_special=True`
4. Map each character's UTF-8 bytes to byte-token IDs via `byte_value_to_id`
5. Apply learned merges in rank order
6. Return the final ID sequence

### decode(ids, *, skip_special_tokens=False)
1. Reject unknown IDs
2. Concatenate canonical token bytes from `id_to_bytes`
3. Decode valid UTF-8 strictly
4. Default behavior (`skip_special_tokens=False`): emit special/reserved token strings verbatim
5. With `skip_special_tokens=True`: omit special/reserved token IDs from output
6. Reserved tokens follow the same contract as special tokens

## Artifact Schema

```json
{"schema_version":"bpe-v1","normalization":"nfc","special_tokens":{"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3},"reserved_tokens":{},"byte_value_to_id":{"0":4,"1":5,...,"255":259},"id_to_bytes":{"4":"00","5":"01",...,"259":"ff"},"vocab":{"<bos>":2,...,"<byte_00>":4,...,"<merge_0>":260,...},"merges":[[97,98,260,0],...],"tokenizer_hash":"sha256hex"}
```

Stored using compact JSON (`sort_keys=True, separators=(",", ":"), ensure_ascii=True`). The same serialization is used for both hashing and storage, guaranteeing byte-identical artifacts from identical training runs.

## Tokenizer Hash

SHA-256 of the canonical JSON payload covering:
- schema_version, normalization
- special_tokens, reserved_tokens
- byte_value_to_id, id_to_bytes
- vocab, merges

Excludes the `tokenizer_hash` field itself (circular reference avoided).

Verified on load: if stored hash does not match recomputed hash, loading fails. The `validate()` method checks all aspects of artifact integrity.

## Artifact Validation

`BPETokenizer.validate()` checks:
1. Schema version is supported
2. Normalization policy is supported
3. Special/reserved maps pass combined validation (no duplicate IDs, strings, or negative IDs)
4. `byte_value_to_id` has exactly keys 0–255 with unique values
5. Byte IDs do not collide with special or reserved IDs
6. `id_to_bytes` is consistent with `byte_value_to_id`
7. All vocabulary IDs are unique
8. Every merge references IDs already defined (by rank order)
9. Merge ranks are unique, contiguous, and zero-based
10. Merge token byte content equals left + right byte concatenation
11. Stored tokenizer hash matches canonical computed hash

Called after training, before save, and during load.

## Save/Load Integrity

- **Save**: write to same-directory temp file with secure random name → reload and verify hash → atomic `rename()`
- **Overwrite**: prohibited by default (`FileExistsError`); use `overwrite=True`
- **Failure safety**: temp file cleaned on failure; existing destination untouched
- **Serialization**: compact JSON (`_compact_serialize()`)
- **Load**: `from_dict()` validates types, schema, normalization, then calls `validate()` for full integrity check
- Unsupported schema versions rejected on load

## Determinism Evidence

Given identical inputs (corpus, vocab_size, special_tokens, seed-independent):
- Identical vocabulary
- Identical merge order and IDs
- Identical byte_value_to_id and id_to_bytes
- Identical tokenizer hash (SHA-256)
- Byte-identical serialized artifact (compact JSON)
- Identical encode results and decoded output
- All verified with 3-second delayed execution test

## Limitations

- Production 64K vocabulary training has NOT been performed and is NOT part of this PR
- Multi-GPU or distributed training is not supported
- Tokenization API is standalone; integration with `BharatTokenizer` is deferred

## Files

| File | Purpose |
|------|---------|
| `bharat/tokenizer/bpe.py` | Core BPE training, encode/decode, serialization, validation |
| `scripts/train_tokenizer.py` | CLI wrapper |
| `tests/tokenizer/test_bpe.py` | 85 unit tests |
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
