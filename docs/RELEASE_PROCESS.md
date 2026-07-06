# Bharat AI Release Process

## Pre-Release Checklist

### 1. Model Validation
- [ ] Overfit one batch (loss → 0)
- [ ] Small-scale training converges (100M tokens)
- [ ] Distributed training test (2+ GPUs)
- [ ] Checkpoint save/load/resume round-trip
- [ ] Tokenizer hash matches checkpoint

### 2. Evaluation
- [ ] All BharatBench modules run
- [ ] No benchmark contamination detected
- [ ] Baseline comparison documented
- [ ] Generation samples reviewed

### 3. Safety
- [ ] Safety evaluation (toxicity, bias, harmful content)
- [ ] PII filter validated
- [ ] Hallucination evaluation
- [ ] Model card written (training data, limitations, intended use)

### 4. Infrastructure
- [ ] CI green on all platforms (3.11, 3.12)
- [ ] Export to HF/safetensors works
- [ ] Export to GGUF/Ollama works
- [ ] API smoke test (health, generate, stream)
- [ ] Auth, rate limiting, CORS configured

### 5. Documentation
- [ ] Model card published on HF Hub
- [ ] Benchmarks recorded in leaderboard
- [ ] Data card published (sources, licence, filters)
- [ ] Known limitations documented
- [ ] Release notes written

## Release Cadence

| Type | Frequency | Bump | Examples |
|------|-----------|------|----------|
| Patch | As needed (bug fixes) | 0.1.0 → 0.1.1 | SFT masking fix, test improvement |
| Minor | Per milestone | 0.1.0 → 0.2.0 | Tokenizer interface, new eval module |
| Major | Per model release | 0.x → 1.0.0 | Bharat-350M, Bharat-1B |

## Release Steps

1. Create release branch: `release/v{version}`
2. Run full checklist
3. Create GitHub release with release notes
4. Push model to HF Hub
5. Update README benchmark section
6. Announce on mailing list / social

## Model Card Template

Each released model includes a `model_card.md` with:
- Model architecture (params, layers, dim, heads, context)
- Training data (sources, size, mixture ratios, licence)
- Tokenizer (type, vocab size, compression ratio, fertility)
- Benchmarks (perplexity, accuracy on all BharatBench tasks)
- Safety evaluation results
- Known limitations
- Intended use and out-of-scope use
- Environmental impact (GPU hours, CO₂)
