# Milestone 4.3 — Local Model Inference Adapter

Milestone 4.3 connects BharatBench prediction generation to a local Bharat checkpoint.

## What this milestone adds

- `LocalInferenceConfig` for local checkpoint prediction settings.
- `LocalCausalLMAdapter` implementing the existing `PredictionAdapter` shape.
- Local checkpoint loading through `BharatForCausalLM.from_pretrained()`.
- Local tokenizer loading through `load_tokenizer()`.
- `scripts/generate_bharatbench_local_predictions.py` for local model prediction JSONL generation.
- Guard tests for local-path-only enforcement and deterministic adapter behaviour.

## What this milestone does not add

- No model training.
- No benchmark downloads.
- No external API calls.
- No scraping.
- No uploads.
- No remote model loading.

## Example

```bash
python scripts/generate_bharatbench_local_predictions.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --checkpoint checkpoints/bharat-smoke \
  --tokenizer tokenizer/tokenizer.json \
  --output predictions.jsonl \
  --max-new-tokens 32 \
  --json

python scripts/run_bharatbench.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --predictions predictions.jsonl \
  --output-dir eval_out \
  --created-at 2026-07-20T00:00:00Z \
  --json
```

## Safety guidance

The local inference adapter accepts only local filesystem paths. It intentionally rejects URLs such as `https://`, `s3://`, `gs://`, and `ftp://` so evaluation cannot silently download models, tokenizers, or data.

Real benchmark evaluation still requires approved local examples and approved local checkpoints. This milestone only connects the adapter interface to local checkpoint inference.
