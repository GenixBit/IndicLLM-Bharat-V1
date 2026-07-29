# Milestone 6.1 — Production Evidence Readiness

This local-only utility converts production-evidence validation into a canonical readiness report for human promotion review.

A manifest is ready for human promotion review only when it:

- passes the strict production-evidence validator;
- has `candidate` status; and
- is not already accepted.

The readiness report does not promote, mutate, upload, or approve evidence. Human review remains required before any candidate can become accepted.

```bash
python scripts/check_production_tokenizer_evidence_readiness.py \
  /path/to/evidence/manifest.json

python scripts/check_production_tokenizer_evidence_readiness.py \
  /path/to/evidence/manifest.json \
  --output /path/to/evidence/readiness.json
```

Exit status is `0` only for a valid candidate ready for human review and `2` otherwise. Output publication is exclusive and refuses to overwrite an existing file.

## Safety boundary

The utility reads caller-selected local files and optionally writes one canonical local JSON report. It performs no training, downloads, external API calls, scraping, uploads, artifact publication, subprocess execution, checkpoint loading, or network access.

This does not provide production tokenizer artifacts and does not close Milestone 6.1.
