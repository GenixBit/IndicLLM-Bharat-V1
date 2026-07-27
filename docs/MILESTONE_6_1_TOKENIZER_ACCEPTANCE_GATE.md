# Milestone 6.1 — Tokenizer Acceptance Gate

**Status:** Draft implementation

This slice adds a deterministic, offline acceptance gate for evaluation reports produced by `scripts/evaluate_tokenizer.py`.

## Scope

The gate validates one named tokenizer against a local JSON threshold configuration. It checks:

- minimum evaluation record count
- required NFC/canonical round-trip pass rate
- maximum unknown-token rate
- complete byte-alphabet coverage when required
- optional aggregate micro-fertility ceiling
- optional per-language micro-fertility ceiling

The result is canonical JSON with a deterministic SHA-256 digest. Failed thresholds return a non-zero CLI exit status. Optional output uses exclusive creation and refuses to overwrite an existing file.

## Local execution

```bash
python scripts/check_tokenizer_acceptance.py \
  --report /local/path/evaluation.json \
  --thresholds configs/tokenizers/bpe-64k-acceptance.json \
  --tokenizer-name bharat-bpe \
  --dry-run
```

To persist the result locally, replace `--dry-run` with `--execute --output /local/path/acceptance.json`.

## Safety boundary

This change does not train a tokenizer, download datasets or benchmarks, call external APIs, scrape data, upload files or artifacts, load model checkpoints, or use the network. Thresholds are initial repository defaults and do not constitute production 64K tokenizer evidence; formal acceptance still requires a caller-selected local production artifact and approved local evaluation corpus.
