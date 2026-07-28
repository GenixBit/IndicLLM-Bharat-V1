# Milestone 6.1 — Synthetic Tokenizer Evidence Pack

## Scope

Synthetic-local-only evidence demonstrating the evaluation and acceptance
pipeline for the deterministic tiny BPE tokenizer.

This is **not** a production evaluation. It uses a 318-token test artifact
against multilingual synthetic fixtures. The acceptance gate correctly
reports failure because the tiny BPE tokenizer cannot achieve full
round-trip fidelity on the multilingual evaluation set.

## Files

| File | Description |
|------|-------------|
| `manifest.json` | Provenance record with all SHA-256 digests |
| `evaluation-report.json` | Evaluation report from `evaluate_tokenizer.py` |
| `acceptance-decision.json` | Acceptance result from `check_tokenizer_acceptance.py` |

## Acceptance result

The committed provisional policy (requires 100% round-trip rate, complete
byte coverage, and >=1 canonical-equivalence example) is **not satisfied**
by the tiny BPE tokenizer. This is expected — the tokenizer has a 318-token
vocabulary that cannot round-trip the full multilingual fixture.

## Determinism

Double-generation confirms byte-identical evaluation reports and acceptance
decisions, stable tokenizer fingerprint, dataset digest, configuration
digests, and acceptance digest.

## Commands used

```
scripts/evaluate_tokenizer.py \
  --tokenizer tests/fixtures/tiny_bpe_tokenizer.json \
  --name tiny-bpe \
  --dataset tests/fixtures/tokenizer_eval/all.jsonl \
  --execute \
  --output-report evidence/tokenizer/milestone-6-1-synthetic/evaluation-report.json

scripts/check_tokenizer_acceptance.py \
  --report evidence/tokenizer/milestone-6-1-synthetic/evaluation-report.json \
  --thresholds configs/tokenizers/bpe-64k-acceptance.json \
  --execute \
  --output evidence/tokenizer/milestone-6-1-synthetic/acceptance-decision.json
```
