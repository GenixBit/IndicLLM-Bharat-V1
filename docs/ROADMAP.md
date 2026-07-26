# Bharat AI Roadmap

## Current State

All current capabilities are powered by the legacy GPT-2 codebase. New development happens under `bharat/`.

## Milestones

### Milestone 1 — Stabilisation ✅

- [x] Repository audit and implementation plan
- [x] Test suite (pytest) and CI (GitHub Actions)
- [x] Linting (ruff) and type checking (mypy)
- [x] Unified tokenizer interface (GPT-2, SentencePiece, HF)
- [x] SFT: assistant-only loss masking
- [x] DPO: per-sample response masking
- [x] Checkpoint metadata (tokenizer hash, git SHA, data version)
- [x] README rebrand and new docs
- [x] Legacy tokenizer references migrated to `bharat/tokenizer/`

### Milestone 2 — Modern Architecture ✅

- [x] RoPE, RMSNorm, SwiGLU, GQA components
- [x] Full Bharat decoder model with KV-cache generation
- [x] Model configurations and sizing calculator (350M, 1B, 3B, 7B)

### Milestone 3 — Data Engine ✅

- [x] Source registry infrastructure with licence validation (Milestone 3.1)
- [x] Quality filters, deduplication, PII detection, pipeline wrapper (Milestone 3.2)
- [x] Data manifests, statistics, shard planning, mixture planning, and contamination checks (Milestone 3.3)
- [x] Manifest/scoring hardening, source-cap redistribution fixes, water-filling algorithm (Milestone 3.3.1–3.3.3)
- [x] Local governed data preparation: file reader, records, shard writer, prepare pipeline, CLI (Milestone 3.4)
- [x] Dataset approval workflow and release packaging (Milestone 3.5)
- [ ] Indic data pipeline unification

### Milestone 4 — Evaluation (BharatBench)

- [x] BharatBench evaluation harness with schema, metrics, runner, reporting, CLI, and tiny fixtures (Milestone 4.1)
- [x] Deterministic local model-to-evaluation prediction adapters and prediction JSONL generator (Milestone 4.2)
- [x] Local model inference adapter for approved checkpoints (Milestone 4.3)
- [x] Language, reasoning, coding, knowledge, safety benchmark category catalog (Milestone 4.4)
- [x] Leaderboard for cross-checkpoint comparison (Milestone 4.5)

### Milestone 5 — Production Serving

- [x] Streaming API foundation: typed events, function specs, local streamer (Milestone 5.1)
- [x] Authentication, rate limiting, metrics (Milestone 5.2)
- [x] Export (safetensors, GGUF F32) — Milestone 5.3 complete

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
