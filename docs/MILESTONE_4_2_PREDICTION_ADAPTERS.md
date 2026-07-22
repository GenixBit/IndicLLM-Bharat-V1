# Milestone 4.2 — Model-to-Evaluation Adapter

Milestone 4.2 adds deterministic local prediction adapters for the BharatBench evaluation harness.

## What this milestone adds

- `PredictionAdapter` protocol for objects that return a prediction string from an `EvalExample`.
- `ExpectedPredictionAdapter` for smoke tests only.
- `EchoPredictionAdapter` for prompt echo testing.
- `ChoiceBaselineAdapter` for deterministic classification baselines.
- `PredictionRunner` for producing exactly one `EvalPrediction` per `EvalExample`.
- Deterministic prediction JSONL writing.
- `scripts/generate_bharatbench_predictions.py` for local prediction-file generation.

## What this milestone does not add

- No model training.
- No benchmark downloads.
- No real model inference.
- No external API calls.
- No scraping.
- No uploads.

## Example

```bash
python scripts/generate_bharatbench_predictions.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --output predictions.jsonl \
  --adapter expected \
  --json

python scripts/run_bharatbench.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --predictions predictions.jsonl \
  --output-dir eval_out \
  --created-at 2026-07-20T00:00:00Z \
  --json
```

## Adapter guidance

The `expected` adapter is intentionally unrealistic and exists only to verify the local evaluation flow. The `choice-baseline` adapter predicts the first available classification choice and is useful as a deterministic baseline. A later milestone can add local checkpoint inference behind the same adapter interface.
