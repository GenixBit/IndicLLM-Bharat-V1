# Milestone 6.1 — Current Implementation Status

**Status:** Active and gated. This document is a current-state reconciliation of the tokenizer work already present in `main`. It does not authorize training, dataset acquisition, external services, uploads, or production artifact handling.

## Implemented offline capabilities

- Deterministic tokenizer-corpus sampling exists in `bharat/tokenizer/sampler.py` with governed release/manifest/approval validation, deterministic selection, exact deduplication, source/language/global caps, provenance records, corpus digests, and local output-path safety checks.
- The corresponding local CLI exists at `scripts/sample_tokenizer_corpus.py` and defaults to dry-run behavior; writing output requires an explicit `--execute` flag.
- The tokenizer training foundation exists in `bharat/tokenizer/train.py` through `train_bpe_tokenizer()` and `train_sentencepiece_tokenizer()`.
- Existing tokenizer metadata, hashing, compatibility validation, and evaluation utilities remain available for later deterministic validation.

## Still not accepted as production evidence

The repository does **not** claim that the Bharat 64K tokenizer has been trained or validated on production data. In particular, the following remain gated:

1. Approved production tokenizer corpus and its authorization record.
2. Final production tokenizer configuration and fixed acceptance thresholds.
3. A reproducible 64K BPE training run using the approved corpus.
4. Extended tokenizer evaluation against the required Indic, English, code, Unicode, and fragmentation criteria.
5. Repeated-run determinism evidence including tokenizer hash stability.
6. Compatibility evidence showing the tokenizer vocabulary size exactly matches the Bharat-350M model configuration.
7. Formal milestone acceptance and promotion evidence.

## Safety boundary

All repository development before the required authorization is available must remain local and deterministic. Do not download training or benchmark datasets, call external APIs, scrape sources, upload data, or upload generated artifacts. Approved local fixtures may be used for software tests.

## Next software step

The next implementation should be a small, local-only capability that consumes explicitly provided configuration/fixtures and verifies a concrete Milestone 6.1 acceptance criterion. Production training and production evidence remain separate controlled activities and must not be simulated by fixture results.
