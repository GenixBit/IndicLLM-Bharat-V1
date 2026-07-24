# Milestone 4.3 — Local Model Inference Adapter

**Status:** Complete

## Objective

Connect the existing BharatBench prediction pipeline to a locally stored
Bharat checkpoint and tokenizer. The implementation remains offline,
safe by default, and rejects all remote paths.

## Implemented

- Local-only path validation for checkpoint, tokenizer, examples, and output.
- `LocalInferenceConfig` and `LocalCausalLMAdapter` APIs.
- A typed `BatchGenerator` protocol for deterministic tests and injected
  generation implementations.
- BharatBench prediction CLI wiring, duplicate-ID validation, and focused
  offline tests using fake generation callables.
- `load_local_causal_lm_adapter()` wires the approved local checkpoint and
  tokenizer paths through the repository's local `BharatForCausalLM`,
  `load_tokenizer()`, and `generate()` APIs without downloading files or
  contacting external services.

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
    BatchGenerator,
    LocalCausalLMAdapter,
    LocalInferenceConfig,
    load_local_causal_lm_adapter,
)
```

`LocalCausalLMAdapter` implements the `PredictionAdapter` protocol and can
be used with `PredictionRunner` or directly through `adapter.predict(example)`.

For tests, generation may be supplied through a lightweight `BatchGenerator`
callable. Production use should provide approved local checkpoint and tokenizer
paths only.
