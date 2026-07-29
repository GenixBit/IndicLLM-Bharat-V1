# Milestone 6.1 — Candidate Evidence Builder

**Status:** In review

This slice assembles a deterministic `candidate` production-tokenizer evidence manifest from caller-provided local files. It does not train a tokenizer, obtain data, download benchmarks, call external services, upload artifacts, or promote evidence to `accepted`.

## Inputs

The builder requires an existing evidence directory containing:

- a deterministic BPE tokenizer artifact
- the exact evaluation input JSONL
- the canonical evaluation report
- the canonical acceptance decision
- the canonical threshold configuration

It derives tokenizer identity, vocabulary size, byte-alphabet completeness, file SHA-256 values, and positive per-language record counts. Every referenced file must resolve inside the evidence root.

## Command

```bash
python scripts/build_production_tokenizer_evidence.py \
  --evidence-root /local/evidence \
  --repository-commit-sha <40-or-64-character-sha> \
  --tokenizer /local/evidence/tokenizer.json \
  --evaluation-input /local/evidence/evaluation.jsonl \
  --evaluation-report /local/evidence/evaluation-report.json \
  --acceptance-decision /local/evidence/acceptance-decision.json \
  --threshold-configuration /local/evidence/thresholds.json \
  --generating-command "<exact local command>" \
  --output /local/evidence/manifest.json
```

The output is canonical JSON, validated through `validate_production_evidence`, published with exclusive no-overwrite semantics, and accompanied by its SHA-256 on stdout.

## Safety boundary

- local files only
- deterministic canonical output
- no overwrite
- no training or model execution
- no dataset or benchmark downloads
- no external APIs, scraping, uploads, or workflow artifacts
- no subprocesses or network access

## Milestone boundary

This is infrastructure for assembling candidate evidence. Milestone 6.1 remains open until a caller-provided production 64K tokenizer package is independently validated and accepted under approved production thresholds.
