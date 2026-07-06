# Bharat AI Roadmap

## Current State

All current capabilities are powered by the legacy GPT-2 codebase. New development happens under `bharat/`.

## Milestones

### Milestone 1 — Stabilisation (Current)

- [x] Repository audit and implementation plan
- [x] Test suite (pytest) and CI (GitHub Actions)
- [x] Linting (ruff) and type checking (mypy)
- [x] Unified tokenizer interface (GPT-2, SentencePiece, HF)
- [x] SFT: assistant-only loss masking
- [x] DPO: per-sample response masking
- [x] Checkpoint metadata (tokenizer hash, git SHA, data version)
- [ ] README rebrand and new docs ← **you are here**
- [ ] Legacy tokenizer references migrated to `bharat/tokenizer/`

### Milestone 2 — Modern Architecture (Next)

- [ ] RoPE, RMSNorm, SwiGLU, GQA components
- [ ] Full Bharat decoder model
- [ ] Model configurations (350M, 1B, 3B, 7B)
- [ ] Legacy GPT-2 moved to `bharat/models/legacy_gpt2.py`

### Milestone 3 — Data Engine

- [ ] Source registry with licence validation
- [ ] Quality filters, deduplication, PII detection
- [ ] Data manifests and contamination checks
- [ ] Indic data pipeline unification

### Milestone 4 — Evaluation (BharatBench)

- [ ] Evaluation runner with registry
- [ ] Language, reasoning, coding, knowledge, safety benchmarks
- [ ] Leaderboard for cross-checkpoint comparison

### Milestone 5 — Production Serving

- [ ] Streaming API with function calling
- [ ] Authentication, rate limiting, metrics
- [ ] Export (safetensors, GGUF)

### Milestone 6 — Bharat-350M Validation

- [ ] 64K BPE tokenizer training and evaluation
- [ ] Smoke test, overfit test, small-scale training
- [ ] Comprehensive benchmark report

### Milestone 7 — Bharat-1B Release

- [ ] Data mixture and compute plan
- [ ] Full pretrain → SFT → DPO pipeline
- [ ] Safety evaluation and model card
- [ ] Public release

## Verification Criteria

Each milestone must pass:
1. All tests (`pytest tests/`) — zero failures
2. Linting (`ruff check .`) — zero errors
3. Type checking (`mypy bharat/`) — zero errors
4. CI green on PR
5. CPU smoke test passes (`python scripts/sanity_check.py`)
6. Documentation reflects actual (not planned) capabilities

## Detailed Implementation Plan

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full PR-by-PR breakdown.
