# OpenCode Implementation Prompt — Milestone 4.3

Implement **Milestone 4.3: Local Model Inference Adapter for approved checkpoints** in `GenixBit/IndicLLM-Bharat-V1`.

## Objective

Connect the existing BharatBench prediction pipeline to a locally stored Bharat checkpoint and tokenizer. The implementation must remain offline, deterministic where possible, and safe by default.

## Required scope

1. Add `bharat/eval/local_inference.py` with:
   - `LocalInferenceConfig` dataclass.
   - `LocalCausalLMAdapter` exposing `predict(example: EvalExample) -> str`.
   - `load_local_causal_lm_adapter(config)`.
   - A typed token-generation protocol or callable wrapper.
2. Reuse existing local code paths:
   - `BharatForCausalLM.from_pretrained()`.
   - `load_tokenizer()`.
   - `generate()`.
3. Add `scripts/generate_bharatbench_local_predictions.py`:
   - Required arguments: `--examples`, `--output`, `--checkpoint`, `--tokenizer`.
   - Optional arguments: `--max-new-tokens`, `--device`, `--json`.
   - Read JSONL examples using the existing `EvalExample` schema.
   - Reject duplicate `example_id` values before loading a model.
   - Write predictions through the existing prediction runner/writer.
4. Export the new adapter APIs from `bharat/eval/__init__.py`.
5. Add a project entry point in `pyproject.toml`.
6. Update `README.md`, `docs/ROADMAP.md`, and add a focused milestone document.

## Mandatory safety boundaries

- No training.
- No dataset or benchmark downloads.
- No external API calls.
- No website scraping.
- No uploads or workflow artifacts.
- No remote model/tokenizer/example/output paths.
- Explicitly reject URL-like paths beginning with `http://`, `https://`, `ftp://`, `s3://`, or `gs://`.
- Model, tokenizer, examples, and output must use local filesystem paths only.

## Tests

Add focused tests covering at minimum:

- Remote checkpoint rejection.
- Remote tokenizer rejection.
- Remote examples rejection.
- Remote output rejection.
- Invalid and negative `max_new_tokens`.
- Empty prompt tokenization rejection.
- Missing checkpoint rejection.
- Missing tokenizer rejection.
- Duplicate example IDs rejected before model loading.
- Adapter decodes only newly generated completion tokens.
- CLI error paths return non-zero status with a clear stderr message.
- CLI subprocess tests must pass `check=False` explicitly.

Use lightweight fake tokenizers and fake generation callables. Do not load a real model in unit tests.

## Formatting and verification

Before opening the implementation PR, run exactly:

```bash
ruff format bharat/eval/__init__.py \
  bharat/eval/local_inference.py \
  scripts/generate_bharatbench_local_predictions.py \
  tests/eval/test_local_inference.py \
  tests/scripts/test_generate_bharatbench_local_predictions.py

ruff check --fix bharat/eval/__init__.py \
  bharat/eval/local_inference.py \
  scripts/generate_bharatbench_local_predictions.py \
  tests/eval/test_local_inference.py \
  tests/scripts/test_generate_bharatbench_local_predictions.py

ruff format --check .
ruff check .
mypy bharat/

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=disabled \
pytest tests/eval/test_local_inference.py \
  tests/scripts/test_generate_bharatbench_local_predictions.py -vv

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WANDB_MODE=disabled \
pytest -m "not slow and not gpu and not integration"

python scripts/validate_data_registry.py
python scripts/validate_data_registry.py --json
python scripts/validate_data_registry.py --strict
git diff --check
```

## Acceptance criteria

- The adapter uses only approved local paths.
- No remote loading or external communication is introduced.
- Prediction JSONL is compatible with the existing BharatBench runner.
- All focused and repository-wide checks pass.
- CI is visibly green.
- Documentation describes actual implemented behavior only.
- Keep the implementation PR scoped; do not add diagnostic workflows or unrelated formatting changes.
