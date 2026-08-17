# Bharat AI Release Process

## Automated Release Packaging & Validation

### 1. Developer Environment Sanity Check
Before cutting a release, verify the native architecture and training loop across target hardware:
```bash
python scripts/sanity_check.py --model bharat --device auto
```

### 2. Build Verified Production Release Bundle
Packages Safetensors weights, GGUF quantized models (Q8_0 / F32), Hugging Face compatible configurations, tokenizer assets, official Model Card, and cryptographic SHA-256 manifest:
```bash
python scripts/build_release_bundle.py \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --model-config configs/models/bharat-350m.yaml \
    --tokenizer data/indic/tokenizer.json \
    --output-dir dist/bharat-350m-v1.0.0 \
    --model-name Bharat-350M \
    --version 1.0.0 \
    --include-gguf \
    --gguf-type Q8_0 \
    --model-card docs/MODEL_CARD.md
```

---

## Pre-Release Checklist

### 1. Model Validation
- [x] Overfit one batch (loss → 0) (`pytest tests/training/test_overfit_350m.py`)
- [x] Checkpoint save/load/resume round-trip bit-exact validation
- [x] Tokenizer hash matches checkpoint metadata

### 2. Evaluation
- [x] All BharatBench modules and categories pass validation (`pytest tests/eval/`)
- [x] Prediction adapters generate verified inference outputs
- [x] Benchmark category catalog and leaderboard generator validated

### 3. Safety & Governance
- [x] License verification and data governance policy checks (`validate_data_registry.py`)
- [x] PII scrubbing, MinHash deduplication, and linguistic quality filters
- [x] Official Model Card written with safety boundaries and carbon accounting (`docs/MODEL_CARD.md`)

### 4. Infrastructure & Serving
- [x] CI green across all unit and integration test suites
- [x] Export to Safetensors verified (`write_safetensors_checkpoint`)
- [x] Export to GGUF (Q8_0 / F32) verified with independent binary parser (`tests/compatibility/test_q8_0_external_gguf.py`)
- [x] Streaming API and rate limiting validated

### 5. Documentation & Packaging
- [x] Official Model Card completed (`docs/MODEL_CARD.md`)
- [x] Release bundle builder validated with SHA-256 manifests (`scripts/build_release_bundle.py`)
- [x] Roadmap and implementation plan updated

---

## Release Cadence

| Type | Frequency | Bump | Examples |
|------|-----------|------|----------|
| Patch | As needed (bug fixes) | 0.1.0 → 0.1.1 | SFT masking fix, test improvement |
| Minor | Per milestone | 0.1.0 → 0.2.0 | Tokenizer interface, pipeline orchestrator |
| Major | Per model release | 0.x → 1.0.0 | Bharat-350M, Bharat-1B |

---

## Release Steps

1. Create release branch: `release/v{version}`
2. Run full checklist: `pytest tests/` and `python scripts/sanity_check.py`
3. Build verified release bundle: `python scripts/build_release_bundle.py ...`
4. Create GitHub release with release notes and attach release manifest
5. Announce on official community channels
