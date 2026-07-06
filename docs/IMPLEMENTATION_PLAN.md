# Bharat AI — Implementation Plan

## Repository Audit Summary

See `docs/CURRENT_STATE_AUDIT.md` for the full audit. Key findings:

- **Verified working:** GPT-2 pretraining, DDP training, basic SFT, basic DPO, evaluation, inference, export, data pipelines
- **Critical defects:** No loss masking in SFT (C1), batch-level prompt length in DPO (C2-C3), tokenizer-embedding size mismatch in SFT (C4)
- **High-severity issues:** Hardcoded `uint16` storage (H1-H3), hardcoded GPT-2 tokenizer (H5-H9), no checkpoint metadata (H10-H12), wildcard CORS (H13)
- **Missing entirely:** Tests, CI, streaming API, authentication, data manifests, deduplication, PII filtering, contamination checks, safety docs, model cards

---

## Target Directory Structure

```
IndicLLM-Bharat-V1/
├── bharat/                       # Main Bharat AI package (NEW)
│   ├── models/                   # Modern architecture
│   ├── tokenizer/                # Unified tokenizer
│   ├── data/                     # Data engine
│   ├── training/                 # Production training
│   ├── posttraining/             # SFT, DPO, alignment
│   ├── evaluation/              # BharatBench
│   ├── serving/                  # Production API
│   ├── agents/                   # Agent framework
│   ├── safety/                   # Safety utilities
│   └── utils/                    # Shared utilities
├── configs/                      # Keep + add Bharat configs
├── train/                        # Keep (legacy)
├── eval/                         # Keep (legacy)
├── inference/                    # Keep (legacy)
├── scripts/                      # Keep
├── infra/                        # Keep
├── docs/                         # NEW documentation
├── tests/                        # NEW test suite
├── model_cards/                  # NEW
├── data_registry/               # NEW
└── .github/workflows/            # NEW CI
```

---

## Milestone Backlog

### Milestone 1 — Stabilisation (PRs 1–6)

**Goal:** Fix critical bugs, unify tokenizer, add tests and CI, make training reproducible.

#### PR 1: Tests + CI infrastructure
- **Files:** `tests/`, `.github/workflows/`, `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml`
- **Content:** Pytest setup, GitHub Actions (lint, type check, test), ruff config
- **Tests:** Skeleton test files for all modules
- **Depends on:** Nothing
- **Rollback:** Remove CI config files
- **Acceptance:** `pytest` runs, `ruff check` passes, CI green on PR

#### PR 2: Unified tokenizer interface
- **Files:** `bharat/tokenizer/` (all 7 files), update `requirements.txt`
- **Content:** Abstract base, SentencePiece/HF loaders, training, evaluation metrics, metadata, normalization
- **Tests:** Round trip, metadata, incompatible tokenizer rejection, uint16/uint32
- **Depends on:** PR 1
- **Rollback:** Delete `bharat/tokenizer/`; existing code uses legacy tokenizer paths
- **Acceptance:** Tokenizer interface loads GPT-2, SentencePiece, and HF tokenizers; metadata round-trips; wrong tokenizer fails clearly

#### PR 3: SFT fix — assistant-only loss masking
- **Files:** `bharat/posttraining/sft.py`, `bharat/posttraining/datasets.py`, `bharat/posttraining/collators.py`, `bharat/posttraining/templates.py`
- **Content:** System/user/assistant roles, multi-turn, loss masking with `-100`, variable-length batching, sequence packing, validation split, checkpoint resume, configurable templates
- **Tests:** Prove user tokens masked, system tokens masked, padding tokens masked, assistant tokens contribute to loss
- **Depends on:** PR 2
- **Rollback:** Restore `train/sft.py`; remove `bharat/posttraining/`
- **Acceptance:** All masking tests pass; loss only on assistant response

#### PR 4: DPO fix — per-sample masking
- **Files:** `bharat/posttraining/dpo.py`, `bharat/posttraining/preference_loss.py`, `bharat/posttraining/preference_dataset.py`
- **Content:** Per-sample response masks, variable-length chosen/rejected, reference model, policy model, reward accuracy, KL monitoring, validation split, checkpoint resume
- **Tests:** Per-sample masking, variable prompt lengths, chosen/rejected logprob correctness
- **Depends on:** PR 2
- **Rollback:** Restore `train/dpo.py`; remove `bharat/posttraining/` files
- **Acceptance:** All masking tests pass; `prompt_len[0]` not used

#### PR 5: Checkpoint metadata + resume
- **Files:** `bharat/training/checkpointing.py`, update checkpoint save/load in legacy and new code
- **Content:** Tokenizer type/hash, git SHA, data version, seed, package versions in checkpoints; optimizer, scheduler, random state recovery
- **Tests:** Checkpoint save/load round trip, resume after interruption, incompatible checkpoint rejection
- **Depends on:** PR 2
- **Rollback:** Revert checkpoint format changes
- **Acceptance:** Checkpoint resume restores exact training state; incompatible checkpoints fail loudly

#### PR 6: README update + new docs
- **Files:** `README.md`, `docs/VISION.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/RELEASE_PROCESS.md`, `docs/GOVERNANCE.md`, `docs/CONTRIBUTING.md`
- **Content:** Rebranded as Bharat AI; clear separation of vision/current/planned; verified results only
- **Tests:** None
- **Depends on:** PR 1
- **Rollback:** Restore old README
- **Acceptance:** No unsupported claims; roadmap clearly separates states

### Milestone 2 — Modern Architecture (PRs 7–9)

**Goal:** RoPE, RMSNorm, SwiGLU, GQA, FlashAttention, modern configs.

#### PR 7: Model components
- **Files:** `bharat/models/config.py`, `bharat/models/rotary.py`, `bharat/models/normalization.py`, `bharat/models/mlp.py`, `bharat/models/attention.py`
- **Content:** `BharatModelConfig` dataclass, RoPE, RMSNorm, SwiGLU, GQA with SDPA/FlashAttention
- **Tests:** Forward pass, backward pass, component-level tests
- **Depends on:** PR 1
- **Rollback:** Delete `bharat/models/` components
- **Acceptance:** Components pass forward/backward; RoPE produces correct rotations

#### PR 8: Full Bharat model
- **Files:** `bharat/models/bharat_model.py`, `bharat/models/generation.py`, `bharat/models/legacy_gpt2.py`
- **Content:** Full decoder model, generation with KV cache, legacy GPT-2 moved to legacy namespace
- **Tests:** Forward pass, backward pass, generation, save/load, legacy checkpoint compatibility
- **Depends on:** PR 7
- **Rollback:** Delete `bharat/models/bharat_model.py`; restore `train/pretrain.py` as sole model
- **Acceptance:** Model trains, generates, saves, loads; legacy checkpoints load correctly

#### PR 9: Model configurations + parameter calculator
- **Files:** `configs/bharat-350m.yaml`, `configs/bharat-1b.yaml`, `configs/bharat-3b.yaml`, `configs/bharat-7b.yaml`, `scripts/calculate_params.py`
- **Content:** Realistic validated configs; exact parameter count script
- **Tests:** Parameter counts match expected; config loading
- **Depends on:** PR 8
- **Rollback:** Delete config files
- **Acceptance:** Parameter calculator matches config estimates within 1%

### Milestone 3 — Data Engine (PRs 10–12)

**Goal:** Versioned, deduplicated, filtered, manifest-tracked data pipeline.

#### PR 10: Source registry + licensing
- **Files:** `bharat/data/registry.py`, `bharat/data/sources.py`, `bharat/data/licensing.py`, `data_registry/`
- **Content:** Data source registration, licence validation, excluded sources, version tracking
- **Tests:** Licence classification, source lookup
- **Depends on:** PR 1
- **Rollback:** Delete files; keep existing data pipelines
- **Acceptance:** Sources with unknown licences are rejected

#### PR 11: Quality filters + deduplication
- **Files:** `bharat/data/language_id.py`, `bharat/data/normalization.py`, `bharat/data/exact_dedup.py`, `bharat/data/fuzzy_dedup.py`, `bharat/data/pii.py`, `bharat/data/quality.py`, `bharat/data/safety_filter.py`
- **Content:** Language identification, Unicode normalization, exact/fuzzy dedup, PII detection, quality scoring, safety filtering
- **Tests:** Dedup correctness, PII detection, quality scoring
- **Depends on:** PR 10
- **Rollback:** Delete files; existing data pipelines unchanged
- **Acceptance:** Dedup removes exact duplicates; PII patterns detected

#### PR 12: Manifests + contamination + sharding
- **Files:** `bharat/data/contamination.py`, `bharat/data/mixture.py`, `bharat/data/sharding.py`, `bharat/data/manifests.py`, `bharat/data/statistics.py`
- **Content:** Benchmark contamination detection, language/domain balancing, sharding with auto uint16/uint32, manifest generation
- **Tests:** Contamination detection, manifest completeness, shard compatibility
- **Depends on:** PR 11
- **Rollback:** Delete files; existing `data/*.py` unchanged
- **Acceptance:** Every shard has a manifest; tokenizer hash tracked; contamination flagged

### Milestone 4 — BharatBench (PRs 13–15)

**Goal:** Comprehensive evaluation framework.

#### PR 13: Evaluation runner + registry
- **Files:** `bharat/evaluation/runner.py`, `bharat/evaluation/registry.py`, `bharat/evaluation/reporting.py`
- **Content:** Benchmark registration, parallel evaluation, JSON + Markdown reporting, leaderboard format
- **Tests:** Registry lookup, report generation
- **Depends on:** PR 1
- **Rollback:** Delete files; keep `eval/benchmark.py`
- **Acceptance:** Reports contain model hash, git commit, tokenizer hash, all generation settings

#### PR 14: Evaluation modules
- **Files:** `bharat/evaluation/language.py`, `bharat/evaluation/reasoning.py`, `bharat/evaluation/coding.py`, `bharat/evaluation/knowledge.py`, `bharat/evaluation/safety.py`, `bharat/evaluation/hallucination.py`, `bharat/evaluation/tool_use.py`, `bharat/evaluation/long_context.py`, `bharat/evaluation/latency.py`, `bharat/evaluation/contamination_check.py`
- **Content:** All evaluation modules with standard benchmark integration
- **Tests:** Each module has at least a smoke test
- **Depends on:** PR 13
- **Rollback:** Delete module files
- **Acceptance:** Language eval runs on Indic datasets; safety eval runs; latency measured

#### PR 15: Leaderboard + reporting
- **Files:** Update `bharat/evaluation/reporting.py` for full leaderboard
- **Content:** Cross-checkpoint comparison, tokenizer comparison, data variant comparison
- **Tests:** Leaderboard generation
- **Depends on:** PR 14
- **Rollback:** Revert reporting changes
- **Acceptance:** Leaderboard compares multiple checkpoints correctly

### Milestone 5 — Serving (PRs 16–18)

**Goal:** Production-ready API with streaming, auth, metrics.

#### PR 16: API refactor
- **Files:** `bharat/serving/api.py`, `bharat/serving/schemas.py`, `bharat/serving/engine.py`, `bharat/serving/batching.py`, `bharat/serving/streaming.py`
- **Content:** Streaming, function calling, structured JSON output, vLLM integration hooks
- **Tests:** API schema validation, streaming response, function calling parsing
- **Depends on:** PR 2, PR 8
- **Rollback:** Revert to legacy `inference/api.py`
- **Acceptance:** Streaming works; function calling works; OpenAI spec compliance

#### PR 17: Auth, rate limiting, metrics
- **Files:** `bharat/serving/authentication.py`, `bharat/serving/rate_limit.py`, `bharat/serving/metrics.py`, `bharat/serving/health.py`, `bharat/serving/safety.py`
- **Content:** API key auth, rate limiting, Prometheus metrics, health/readiness, safe defaults, configurable CORS
- **Tests:** Auth rejection, rate limit enforcement, metrics endpoint
- **Depends on:** PR 16
- **Rollback:** Revert to legacy `inference/api.py`
- **Acceptance:** Unauthenticated requests rejected; CORS configurable; metrics at `/metrics`

#### PR 18: Export + quantization
- **Files:** `bharat/serving/export.py` (new), update `inference/export_ollama.py`
- **Content:** safetensors export, GGUF/Ollama export, quantization support
- **Tests:** Export round trip
- **Depends on:** PR 8
- **Rollback:** Revert to legacy export
- **Acceptance:** Models export to safetensors and GGUF correctly

### Milestone 6 — Bharat-350M Validation (PRs 19–20)

**Goal:** First validated Bharat model.

#### PR 19: Tokenizer training + validation
- **Files:** `bharat/tokenizer/train.py`, `bharat/tokenizer/evaluate.py`
- **Content:** Train 64K BPE tokenizer on multilingual Indic + English + code data; evaluate compression, fertility, code efficiency
- **Tests:** Tokenizer evaluation metrics, language-wise fertility
- **Depends on:** PR 2
- **Rollback:** Revert tokenizer training
- **Acceptance:** Tokenizer compression ratio is better than GPT-2 on Indic languages

#### PR 20: Bharat-350M smoke test + benchmark
- **Files:** Training configs, evaluation configs
- **Content:** Overfit-one-batch test, small-scale training (100M tokens), benchmark report
- **Tests:** Overfit test, distributed training test
- **Depends on:** PR 9, PR 15, PR 19
- **Rollback:** None (doesn't overwrite)
- **Acceptance:** Model overfits one batch; distributed training converges; benchmark report generated

### Milestone 7 — Bharat-1B (PRs 21–22)

**Goal:** Full-scale Bharat-1B release candidate.

#### PR 21: Data mixture + compute plan
- **Files:** `bharat/data/mixture.py`, compute plan document
- **Content:** Final data mixture ratios, licence review, compute budget, scaling laws validation
- **Tests:** Mixture verification
- **Depends on:** PR 12
- **Rollback:** None
- **Acceptance:** Mixture is licensed, balanced, and documented

#### PR 22: Bharat-1B training + evaluation + model card
- **Files:** Training logs, benchmark results, model card
- **Content:** Full pretraining → SFT → DPO pipeline; comprehensive evaluation; safety review; model card
- **Tests:** Full evaluation suite
- **Depends on:** PR 20, PR 21
- **Rollback:** None
- **Acceptance:** All benchmark results documented; safety review passed; model card complete

---

## Pull Request Sequence

| Order | PR Title | Main Files | Depends On | Duration |
|-------|----------|-----------|------------|----------|
| 1 | ci: Add tests, linting, and CI infrastructure | `tests/`, `.github/`, `pyproject.toml`, `ruff.toml` | — | 1-2 days |
| 2 | feat: Unified tokenizer interface | `bharat/tokenizer/` (7 files) | PR 1 | 2-3 days |
| 3 | fix: SFT assistant-only loss masking | `bharat/posttraining/sft.py`, `datasets.py`, `collators.py`, `templates.py` | PR 2 | 2-3 days |
| 4 | fix: DPO per-sample response masking | `bharat/posttraining/dpo.py`, `preference_loss.py`, `preference_dataset.py` | PR 2 | 2-3 days |
| 5 | feat: Checkpoint metadata and resume | `bharat/training/checkpointing.py`, updates to save/load | PR 2 | 1-2 days |
| 6 | docs: Rebrand as Bharat AI | `README.md`, `docs/VISION.md`, `docs/ROADMAP.md`, etc. | PR 1 | 1 day |
| 7 | feat: Model components (RoPE, RMSNorm, SwiGLU, GQA) | `bharat/models/config.py`, `rotary.py`, `normalization.py`, `mlp.py`, `attention.py` | PR 1 | 3-4 days |
| 8 | feat: Bharat model + legacy GPT-2 move | `bharat/models/bharat_model.py`, `generation.py`, `legacy_gpt2.py` | PR 7 | 2-3 days |
| 9 | feat: Model configs + parameter calculator | `configs/bharat-*.yaml`, `scripts/calculate_params.py` | PR 8 | 1 day |
| 10 | feat: Data source registry and licensing | `bharat/data/registry.py`, `sources.py`, `licensing.py`, `data_registry/` | PR 1 | 2 days |
| 11 | feat: Data quality filters and deduplication | `bharat/data/*_dedup.py`, `pii.py`, `quality.py`, `safety_filter.py` | PR 10 | 2-3 days |
| 12 | feat: Data manifests and contamination checks | `bharat/data/contamination.py`, `manifests.py`, `sharding.py` | PR 11 | 2 days |
| 13 | feat: Evaluation runner and reporting | `bharat/evaluation/runner.py`, `registry.py`, `reporting.py` | PR 1 | 2 days |
| 14 | feat: Evaluation modules | `bharat/evaluation/*.py` (12 modules) | PR 13 | 3-4 days |
| 15 | feat: Leaderboard and comparison reporting | Updates to `reporting.py` | PR 14 | 1-2 days |
| 16 | feat: Streaming API and function calling | `bharat/serving/api.py`, `schemas.py`, `engine.py`, `streaming.py` | PR 2, 8 | 2-3 days |
| 17 | feat: Auth, rate limiting, metrics | `bharat/serving/authentication.py`, `rate_limit.py`, `metrics.py` | PR 16 | 1-2 days |
| 18 | feat: Export improvements (safetensors, quantization) | `bharat/serving/export.py` | PR 8 | 1-2 days |
| 19 | feat: Bharat 64K tokenizer training and validation | `bharat/tokenizer/train.py`, `evaluate.py` | PR 2 | 2-3 days |
| 20 | test: Bharat-350M smoke test and benchmark | Training/eval configs | PR 9, 15, 19 | 1-2 weeks |
| 21 | feat: Bharat-1B data mixture and compute plan | `bharat/data/mixture.py`, plan doc | PR 12 | 1 week |
| 22 | feat: Bharat-1B training, evaluation, and release | Training, evaluation, model card | PR 20, 21 | 4-8 weeks |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|-----------|--------|------------|-------|
| Tokenizer incompatibility breaks existing checkpoints | Medium | High | Store tokenizer hash; validate on load; keep legacy GPT-2 path | Tokenizer lead |
| SFT/DPO fixes change training dynamics requiring retuning | Medium | Medium | Validate against small datasets; compare loss curves | Post-training lead |
| Modern model doesn't match GPT-2 quality at small scale | Medium | Medium | Run ablation on 10M-scale before full 350M training | Model architecture lead |
| Data pipeline changes break existing binary shards | Low | High | Version shard format; backward-compatible loader | Data lead |
| GPU costs exceed budget for 350M validation | Medium | Medium | Start with CPU/MPS for smoke tests; only GPU for final validation | Infrastructure lead |
| Benchmark contamination leads to overestimated capability | Medium | High | Add contamination detection before any release | Evaluation lead |
| Safety issues in released model | Low | High | Comprehensive safety eval; documented limitations; staged release | Safety lead |
| Insufficient Indic language data for 64K tokenizer | Medium | Medium | Evaluate fertility per language; augment with Romanized and Hinglish data | Data lead |
| Catastrophic forgetting during SFT/DPO | Medium | High | Track reference KL; use validation-based early stopping | Post-training lead |

---

## Acceptance Criteria — Milestone 1

1. **All tests pass** — `pytest` completes with zero failures
2. **Linting passes** — `ruff check .` completes clean
3. **Type checking passes** — `mypy` or `pyright` on all `bharat/` code
4. **CI is green** — GitHub Actions passes for lint, type check, test
5. **Unified tokenizer used everywhere** — pretraining, SFT, DPO, evaluation, inference, API, export all load through `bharat/tokenizer/`
6. **SFT loss is assistant-only** — verified by test: user, system, padding tokens have `-100` labels and don't contribute to loss
7. **DPO uses per-sample masks** — verified by test: variable-length prompts don't use `prompt_len[0]`
8. **Checkpoints include metadata** — tokenizer type, tokenizer hash, vocab_size, git SHA, data version, seed all present
9. **Incompatible tokenizer rejected** — loading a checkpoint with wrong tokenizer fails with clear error message
10. **Training is restartable** — checkpoint resume restores optimizer, scheduler, and random state
11. **CPU smoke test passes** — `scripts/sanity_check.py` completes
12. **API has configurable CORS** — default is restrictive, not wildcard
13. **README has no unsupported claims** — vision section clearly separated from current capabilities
14. **Roadmap is accurate** — completed/active/planned states clearly indicated
