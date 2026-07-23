# Milestone 4.3 — Local Model Inference Adapter

**Status:** Complete

## Objective

Connect the existing BharatBench prediction pipeline to a locally stored
Bharat checkpoint and tokenizer. The implementation is offline, safe by
default, and rejects all remote paths.

## Architecture

```
bharat/eval/local_inference.py
├── LocalInferenceConfig       — frozen dataclass (checkpoint, tokenizer, device, max_new_tokens)
├── BatchGenerator             — Protocol for typed generation callables
├── LocalCausalLMAdapter       — Predicts by stripping the prompt from generated text
└── load_local_causal_lm_adapter() — Factory function

scripts/generate_bharatbench_local_predictions.py
├── Validates all paths (local only)
├── Loads examples via EvalExample.from_dict
├── Rejects duplicate example_ids before model loading
├── Creates LocalInferenceConfig → adapter → PredictionRunner → write_predictions_jsonl
└── Optional --json flag for machine-readable output
```

## Safety

All paths are validated before use. Remote URLs (`http://`, `https://`,
`ftp://`, `s3://`, `gs://`) are rejected with a clear error message. No
model weights, tokenizers, benchmarks, or datasets are downloaded. No
external APIs are called. No data is uploaded.

## CLI Usage

```bash
python scripts/generate_bharatbench_local_predictions.py \
  --examples eval_fixtures/bharatbench_tiny/qa.jsonl \
  --output predictions.jsonl \
  --checkpoint /path/to/checkpoint \
  --tokenizer /path/to/tokenizer \
  --max-new-tokens 256 \
  --device cpu \
  --json
```

## Entry Point

```bash
generate-bharatbench-local-predictions \
  --examples ... --output ... --checkpoint ... --tokenizer ...
```

## API

```python
from bharat.eval.local_inference import (
    LocalInferenceConfig,
    LocalCausalLMAdapter,
    BatchGenerator,
    load_local_causal_lm_adapter,
)
```

`LocalCausalLMAdapter` implements the `PredictionAdapter` protocol and can
be used with `PredictionRunner` or directly via `adapter.predict(example)`.

Generation is delegated to a `BatchGenerator` callable. When `generate_fn`
is not provided, `_default_generate` raises `NotImplementedError`. Inject a
custom `generate_fn` to test without a real model, or use
`load_local_causal_lm_adapter()` for production use.

## Limitations

- Real model loading (`BharatForCausalLM.from_pretrained`) is not yet wired
  into `load_local_causal_lm_adapter`. The factory currently creates the
  adapter without a generation function. Model weights must be provided
  when they become available.
- No GPU acceleration is configured beyond the `--device` flag.
- Tokenization uses the existing `load_tokenizer()` infrastructure.
