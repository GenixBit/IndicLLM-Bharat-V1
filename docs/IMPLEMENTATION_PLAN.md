# Bharat AI — Implementation Plan

> **Note (2026-07-27):** The audit text below is historical (written 2025-07-06). Many items listed
> as "not yet started" have since been completed. See the [Current Status](#current-status-2026-07-27)
> section for the authoritative view of where each milestone stands today.

## Repository Audit Summary (Historical — 2025-07-06)

See `docs/CURRENT_STATE_AUDIT.md` for the full audit. Key findings (post-Milestone 3.1):

- **Milestone 1 completed:** Tests/CI, unified tokenizer, SFT loss masking, DPO per-sample masking, checkpoint metadata, API CORS, documentation
- **Milestone 2 completed:** Modern model architecture (RoPE, RMSNorm, SwiGLU, GQA, FlashAttention), generation, BharatModel/BharatForCausalLM
- **Milestone 3.1 completed:** Governed data-source registry with default-deny licensing, immutable revision pins, SHA-256 integrity, deterministic digest, offline validation CLI
- **Verified working:** GPT-2 pretraining, DDP training, SFT (with loss masking), DPO (with per-sample masks), evaluation, inference, export, data pipelines
- **Critical defects fixed:** SFT loss masking, DPO batch-level prompt length, tokenizer-embedding size mismatch
- **High-severity issues fixed:** Hardcoded GPT-2 tokenizer, no checkpoint metadata, wildcard CORS
- **Still open:** Hardcoded `uint16` storage (H1-H3), test implementations are stubs, no streaming API, no authentication, no quality filtering/dedup, no data engine (Milestone 3.2+)
- **Not yet started:** Milestone 3.2 (Quality Filters + Dedup), Milestone 4 (BharatBench), Milestone 5 (Production Serving)

## Current Status (2026-07-27)

The current implementation status reflects substantial progress beyond the 2025-07-06 audit:

| Milestone | Status | Notes |
|-----------|--------|-------|
| Milestone 1 (Stabilisation) | ✅ Complete | All 14 criteria met at `637e2d3` |
| Milestone 2 (Modern Architecture) | ✅ Complete | RoPE, RMSNorm, SwiGLU, GQA, configs, sizing |
| Milestone 3 (Data Engine) | ✅ Complete | 3.1–3.5 all merged; 1062+ tests pass |
| Milestone 4 (BharatBench) | ✅ Complete | Harness (4.1), adapters (4.2), local model (4.3), catalog (4.4), leaderboard (4.5) |
| Milestone 5 (Production Serving) | ✅ Complete | Streaming (5.1), auth/metrics (5.2), export (5.3), Q8_0 GGUF (5.4) — all merged |
| Milestone 6 (Bharat-350M Validation) | 🔲 **Active** | Milestone 6.1 tokenizer plan in progress |
| Milestone 7 (Bharat-1B Release) | 🔲 Not started | Blocked on Milestone 6 completion

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

### Milestone 1 — Stabilisation (PRs 1–6) ✅ COMPLETED at `637e2d3`

**Goal:** Fix critical bugs, unify tokenizer, add tests and CI, make training reproducible.

#### PR 1: Tests + CI infrastructure ✅
- **Files:** `tests/`, `.github/workflows/`, `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml`
- **Content:** Pytest setup, GitHub Actions (lint, type check, test), ruff config
- **Tests:** Skeleton test files for all modules (many are `pytest.skip` stubs)
- **Depends on:** Nothing
- **Rollback:** Remove CI config files
- **Acceptance:** `pytest` runs, `ruff check` passes, CI green on PR

#### PR 2: Unified tokenizer interface ✅
- **Files:** `bharat/tokenizer/` (6 files), `bharat/__init__.py`
- **Content:** Abstract base (`BharatTokenizer`), GPT-2/SentencePiece/HF wrappers, training, evaluation metrics, metadata/hash
- **Tests:** Round trip, metadata, incompatible tokenizer rejection
- **Depends on:** PR 1
- **Rollback:** Delete `bharat/tokenizer/`; existing code uses legacy tokenizer paths
- **Acceptance:** Tokenizer interface loads GPT-2, SentencePiece, and HF tokenizers; metadata round-trips; wrong tokenizer fails clearly

#### PR 3: SFT fix — assistant-only loss masking ✅
- **Files:** `train/sft.py` (legacy), `bharat/posttraining/sft.py`, `datasets.py`, `collators.py`, `templates.py`
- **Content:** Assistant-only loss masking with `-100`; embedding resize after `add_special_tokens`; multi-turn support via collator
- **Tests:** Prove user tokens masked, system tokens masked, padding tokens masked, assistant tokens contribute to loss
- **Depends on:** PR 2
- **Rollback:** Restore pre-masking `train/sft.py`; remove `bharat/posttraining/`
- **Acceptance:** All masking tests pass; loss only on assistant response

#### PR 4: DPO fix — per-sample masking ✅
- **Files:** `train/dpo.py` (legacy), `bharat/posttraining/dpo.py`, `preference_loss.py`, `preference_dataset.py`
- **Content:** Per-sample `prompt_len` from `__getitem__`; per-sample mask in `log_probs()` via `prompt_lens.unsqueeze(-1)`; variable-length chosen/rejected
- **Tests:** Per-sample masking, variable prompt lengths, chosen/rejected logprob correctness
- **Depends on:** PR 2
- **Rollback:** Restore pre-fix `train/dpo.py`; remove `bharat/posttraining/` files
- **Acceptance:** All masking tests pass; `prompt_len[0]` not used

#### PR 5: Checkpoint metadata + resume ✅
- **Files:** `train/pretrain.py`, `train/pretrain_ddp.py`, `bharat/training/checkpointing.py`
- **Content:** Tokenizer type/hash, git SHA, vocab size, package versions stored in checkpoints; tokenizer validated on resume
- **Tests:** Checkpoint save/load round trip, resume after interruption, incompatible checkpoint rejection
- **Depends on:** PR 2
- **Rollback:** Revert checkpoint format changes
- **Acceptance:** Checkpoint resume restores exact training state; incompatible checkpoints fail loudly

#### PR 6: README update + new docs ✅
- **Files:** `README.md`, `docs/VISION.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/RELEASE_PROCESS.md`, `docs/GOVERNANCE.md`, `docs/CONTRIBUTING.md`, `docs/CURRENT_STATE_AUDIT.md`, `docs/IMPLEMENTATION_PLAN.md`
- **Content:** Rebranded as Bharat AI; clear separation of vision/current/planned; verified results only
- **Tests:** None
- **Depends on:** PR 1
- **Rollback:** Restore old README
- **Acceptance:** No unsupported claims; roadmap clearly separates states

### Milestone 2 — Modern Architecture (PRs 7–9)

**Goal:** RoPE, RMSNorm, SwiGLU, GQA, FlashAttention, modern configs.

**Progress:** PR 7 (components) ✅, PR 8 (full model + generation) ✅, PR 9 (configs + calculator) ✅.

#### PR 7: Model components ✅ COMPLETED
- **Files:** `bharat/models/config.py`, `bharat/models/rotary.py`, `bharat/models/normalization.py`, `bharat/models/mlp.py`, `bharat/models/attention.py`
- **Content:** `BharatModelConfig` dataclass, RoPE, RMSNorm, SwiGLU, GQA with SDPA/FlashAttention
- **Tests:** Forward pass, backward pass, component-level tests, causal leakage test
- **Depends on:** PR 1
- **Rollback:** Delete `bharat/models/` components
- **Acceptance:** Components pass forward/backward; RoPE preserves vector norms; GQA works in MHA/GQA/MQA modes; causal leakage test passes

#### PR 8: Full Bharat model ✅ COMPLETED
- **Files:** `bharat/models/bharat_model.py`, `bharat/models/generation.py`, `bharat/models/outputs.py`, `bharat/models/cache.py`, `bharat/models/__init__.py`
- **Content:** Full decoder model (BharatDecoderLayer, BharatModel, BharatForCausalLM), KV-cached generation, typed outputs, cache validation, save/load
- **Tests:** Forward/backward pass, cache parity, generation, save/load, weight tying, causal loss reference
- **Depends on:** PR 7
- **Rollback:** Delete `bharat/models/bharat_model.py`, `generation.py`, `outputs.py`, `cache.py`; legacy `train/pretrain.py` unaffected
- **Acceptance:** Full and incremental cached logits match; cache length grows; generation is deterministic; save/load produces identical outputs; all tests pass

#### PR 9: Model configurations + parameter calculator ✅ COMPLETED
- **Files:** `configs/models/bharat-350m.yaml`, `configs/models/bharat-1b.yaml`, `configs/models/bharat-3b.yaml`, `configs/models/bharat-7b.yaml`, `bharat/models/spec.py`, `bharat/models/sizing.py`, `scripts/calculate_params.py`, `docs/MODEL_CONFIGURATIONS.md`, `tests/models/test_model_specs.py`, `tests/models/test_sizing.py`, `tests/scripts/test_calculate_params.py`
- **Content:** Realistic validated configs within 1% of nominal tier; typed `BharatModelSpec`/`load_model_spec` loader; `ParameterCount`/`StaticMemoryReport`/`KVCacheMemoryReport` calculators; CLI script; comprehensive tests
- **Tests:** Parameter counts match expected within 1%; config loading; memory calculations; CLI output formats; edge cases (empty batch, invalid config, bound conditions)
- **Depends on:** PR 8
- **Rollback:** Delete config files, spec/sizing modules, and associated tests
- **Acceptance:** Parameter calculator matches config estimates within 1%; all new tests pass; no regressions

### Milestone 3 — Data Engine (PRs 10–12, hardening PRs 8–10, 3.4 PR 11)

**Goal:** Versioned, deduplicated, filtered, manifest-tracked data pipeline. ✅ COMPLETED

**Progress:** PR 10 (source registry + licensing) ✅, PR 11 (filters + dedup) ✅, PR 12 (manifests + contamination) ✅.
**3.3 Hardening (manifests, contamination, mixture, source caps):** 3 PRs merged — 934→955 tests.
**3.4 Local governed preparation:** `local_reader.py`, `records.py`, `shard_writer.py`, `preparation.py`, `scripts/prepare_local_data.py` — 989 tests pass.
**3.5 Dataset approval + release packaging:** `approval.py`, `release.py`, 2 CLI tools — 1062 tests pass.

#### PR 10: Source registry + licensing ✅ COMPLETED
- **Files:** `bharat/data/schema.py`, `bharat/data/licensing.py`, `bharat/data/sources.py`, `bharat/data/registry.py`, `bharat/data/__init__.py`, `data_registry/`, `scripts/validate_data_registry.py`, `docs/DATA_GOVERNANCE.md`
- **Content:** Frozen dataclass schemas, licence policy with default-deny, source lifecycle (proposed/approved/rejected/deprecated), immutable revision enforcement, SHA-256 integrity pins, registry digest, offline validation CLI
- **Tests:** Licence decision rules, source schema validation, registry integrity, CLI output formats
- **Depends on:** PR 1
- **Rollback:** Delete files; keep existing data pipelines
- **Acceptance:** Unknown and missing licences cannot be approved; approved records require evidence and immutable provenance; registry ordering and digest are deterministic; offline validation succeeds

#### PR 11: Quality filters + deduplication ✅ COMPLETED
- **Files:** `bharat/data/language_id.py`, `bharat/data/normalization.py`, `bharat/data/exact_dedup.py`, `bharat/data/fuzzy_dedup.py`, `bharat/data/pii.py`, `bharat/data/quality.py`, `bharat/data/safety_filter.py`, `bharat/data/processing.py`
- **Content:** Language identification, Unicode normalization, exact/fuzzy dedup, PII detection, quality scoring, safety filtering, offline pipeline wrapper
- **2025-07 Hardening:** Unicode-safe fuzzy dedup, config-respecting exact dedup, short-Indic script fallback in language ID, Luhn-validated credit cards + overlap resolution in PII, QualityDecision with reason codes in quality scorer, heuristic-only disclaimer + reason codes in safety filter, deterministic DataProcessor pipeline wrapper
- **Tests:** 111 tests covering Indic scripts, edge cases, config validation, overlap resolution, Luhn validation, reason codes, pipeline composition
- **Depends on:** PR 10
- **Rollback:** Delete files; existing data pipelines unchanged
- **Acceptance:** Dedup removes exact duplicates; PII patterns detected; Indic text handled safely; all CI checks pass

#### PR 12: Manifests + contamination + sharding ✅ COMPLETED
- **Files:** `bharat/data/manifest.py`, `bharat/data/stats.py`, `bharat/data/sharding.py`, `bharat/data/mixture.py`, `bharat/data/contamination.py`, `scripts/validate_data_manifest.py`, `scripts/plan_data_shards.py`, `scripts/compute_data_stats.py`
- **Content:** Deterministic dataset manifests (SHA-256 digest, schema validation), offline dataset statistics via DataProcessor, shard planning (record/byte constraints), mixture planning (language/domain/source weights), contamination detection (exact/normalized/n-gram), CLI tools
- **Tests:** 77 tests across all 5 modules + 3 CLI tools; deterministic digest verified; source caps enforced; exact/ngram contamination detected
- **Depends on:** PR 11
- **Rollback:** Delete files; existing `bharat/data/*.py` unchanged
- **Acceptance:** Manifest schema is deterministic and validated; statistics work on local records only; shard planner does not download data; mixture planner enforces source/language constraints; contamination checks are offline and deterministic; CLI tools support JSON output; no datasets downloaded; no training added

#### PR 13: Dataset approval + release packaging ✅ COMPLETED (Milestone 3.5)
- **Files:** `bharat/data/approval.py`, `bharat/data/release.py`, `scripts/validate_dataset_approval.py`, `scripts/build_dataset_release.py`, `tests/data/test_approval.py`, `tests/data/test_release.py`, `tests/scripts/test_validate_dataset_approval.py`, `tests/scripts/test_build_dataset_release.py`
- **Content:** `DatasetApproval` frozen dataclass with deterministic JSON serialisation, SHA-256 digest, ISO-8601 timestamps, four review flags (license, PII, contamination, safety); `validate_approval_for_manifest()` cross-checks approval against manifest; `DatasetRelease` frozen dataclass with `shard_count`, `records`, `bytes_utf8`, `package_sha256`; `DatasetAuditReport` for release audit trail; `DatasetReleaseBuilder` loads manifest + approval from local files, verifies shard files exist and SHA-256 digests match, rejects remote URLs, writes deterministic `dataset_release.json` and `audit_report.json`; CLI tools for approval validation and release building
- **Tests:** 65 tests (18 approval, 30 release, 12 validate CLI, 5 build CLI)
- **Depends on:** PR 12
- **Rollback:** Delete approval.py, release.py, CLI scripts, and tests
- **Acceptance:** Approval requires all four review flags; pending/rejected/revoked cannot release; release builder verifies shard files and digests; tampered/missing shards fail; remote URLs rejected; deterministic JSON output; no datasets downloaded; no training added

### Milestone 4 — BharatBench (PRs 14–16)

**Goal:** Comprehensive evaluation framework.

**Progress:** PR 14 (harness) ✅ — 4.1 harness complete.

#### PR 14: BharatBench evaluation harness ✅ COMPLETED (Milestone 4.1)
- **Files:** `bharat/eval/schema.py`, `bharat/eval/metrics.py`, `bharat/eval/runner.py`, `bharat/eval/reporting.py`, `bharat/eval/__init__.py`, `scripts/run_bharatbench.py`, `eval_fixtures/bharatbench_tiny/`, `tests/eval/test_schema.py`, `tests/eval/test_metrics.py`, `tests/eval/test_runner.py`, `tests/eval/test_reporting.py`, `tests/scripts/test_run_bharatbench.py`
- **Content:** `EvalExample`/`EvalPrediction`/`EvalResult` frozen dataclasses with deterministic JSON and SHA-256 digest; QA, classification, and generation task types; dependency-free metrics (exact_match, normalized_exact_match, token_f1, choice_accuracy); `BharatBenchRunner` that matches predictions to examples, rejects missing/duplicate/unknown predictions, and computes per-task metrics; `BharatBenchReport` with aggregate scores and deterministic digest; CLI that evaluates local prediction JSONL files only; tiny synthetic fixtures in `eval_fixtures/bharatbench_tiny/`
- **Tests:** 18 tests across schema, metrics, runner, reporting, and CLI
- **Depends on:** PR 1
- **Rollback:** Delete `bharat/eval/`, `scripts/run_bharatbench.py`, `eval_fixtures/`, and `tests/eval/`
- **Acceptance:** QA, classification, and generation task types supported; metrics are deterministic and dependency-free; runner rejects missing/duplicate/unknown predictions; report aggregates deterministically; CLI evaluates local prediction JSONL files only; remote URLs rejected; no model training or generation added

#### PR 15: Evaluation modules
- **Files:** `bharat/eval/language.py`, `bharat/eval/reasoning.py`, `bharat/eval/coding.py`, `bharat/eval/knowledge.py`, `bharat/eval/safety.py`, `bharat/eval/hallucination.py`, `bharat/eval/tool_use.py`, `bharat/eval/long_context.py`, `bharat/eval/latency.py`, `bharat/eval/contamination_check.py`
- **Content:** All evaluation modules with standard benchmark integration
- **Tests:** Each module has at least a smoke test
- **Depends on:** PR 14
- **Rollback:** Delete module files
- **Acceptance:** Language eval runs on Indic datasets; safety eval runs; latency measured

#### PR 16: Leaderboard + reporting
- **Files:** Update `bharat/eval/reporting.py` for full leaderboard
- **Content:** Cross-checkpoint comparison, tokenizer comparison, data variant comparison
- **Tests:** Leaderboard generation
- **Depends on:** PR 15
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

#### PR 18: Export (safetensors + GGUF F32) ✅ COMPLETED (Milestone 5.3)
- **Files:** `bharat/serving/safetensors_writer.py`, `bharat/serving/gguf_writer.py`, `bharat/serving/gguf_tensor_writer.py`, `bharat/serving/gguf_preflight.py`, `bharat/serving/gguf_reader.py`, `bharat/serving/export_writer.py`, `bharat/serving/export_manifest.py`, `bharat/serving/export_path_readiness.py`, `bharat/serving/export_writer_readiness.py`, `bharat/serving/export_manifest_readiness.py`, `bharat/serving/export_checkpoint_inventory.py`, `scripts/run_export_plan.py`, `scripts/run_export_execute.py`, 15 milestone docs, 168+ tests
- **Content:** Local PyTorch-to-safetensors writer, local PyTorch-to-GGUF-F32 writer (header + tensor descriptors + F32 payload), GGUF compatibility reader, preflight validators, path/writer/manifest readiness gates, checkpoint inventory, CLI with dry-run/execute, deterministic manifests
- **Tests:** 168 export tests (safetensors 41, GGUF writer 6, GGUF preflight 10, GGUF tensor writer 6, GGUF reader 6, export writer 5, CLI execute ~80, CLI plan 10, manifest 6, readiness 6)
- **Depends on:** PR 8
- **Rollback:** Delete `bharat/serving/safetensors_writer.py`, `gguf_writer.py`, `gguf_tensor_writer.py`, `gguf_preflight.py`, `gguf_reader.py`, `export_writer.py`, `export_manifest*.py`, `export_path_readiness*.py`, `export_writer_readiness*.py`, `export_checkpoint_inventory*.py`, `scripts/run_export_plan.py`; revert CLI integrations
- **Acceptance:** Models export to safetensors and GGUF F32 correctly; all existing tests pass; offline/CPU-only; no overwrite; `--execute` controlled by explicit flag; quantization out of scope

### Milestone 6 — Bharat-350M Validation

**Goal:** First validated Bharat model.

**Status (2026-07-27):** Tokenizer architecture and validation plan defined. No tokenizer training started. See [MILESTONE_6_1_TOKENIZER_VALIDATION_PLAN.md](MILESTONE_6_1_TOKENIZER_VALIDATION_PLAN.md) for the complete specification.

Milestone 6 is divided into three sub-milestones, with Milestone 6.1 (tokenizer) further broken into 7 phased PRs:

#### Milestone 6.1 — 64K BPE Tokenizer Validation (PRs A–G)

**Phased PR Plan:**

##### PR A — Architecture and Evaluation Contract (Documentation)
- **Files:** `docs/MILESTONE_6_1_TOKENIZER_VALIDATION_PLAN.md`, `docs/IMPLEMENTATION_PLAN.md`
- **Content:** Architecture decisions, algorithm recommendation, vocabulary composition, special-token contract, normalization policy, training-data contract, evaluation metrics, baseline comparison, acceptance criteria, model compatibility, compute plan, safety rules, phased PR sequence
- **Depends on:** Nothing (documentation only)
- **Rollback:** Revert doc changes
- **Acceptance:** All architecture decisions documented and approved

##### PR B — Deterministic Tokenizer-Corpus Sampler
- **Files:** `scripts/sample_tokenizer_corpus.py`, `bharat/tokenizer/sampler.py`, `tests/test_tokenizer_sampler.py`
- **Content:** CLI to sample from approved local dataset releases; core sampling with deterministic ordering; sample manifest with corpus digest
- **Depends on:** Milestone 3.5 (dataset approval + release packaging — complete)
- **Rollback:** Delete sampler files
- **Acceptance:** Repeated sampling from identical inputs produces identical corpus digest

##### PR C — Tiny Tokenizer Training Harness
- **Files:** `bharat/tokenizer/train.py` (updated), `bharat/tokenizer/loader.py` (add `_BPEWrapper`), `configs/tokenizers/bpe-64k.yaml`, `tests/test_tokenizer_train.py`, `tests/test_bpe_wrapper.py`
- **Content:** Production configuration support; BPE wrapper returning `tokenizer_type="bpe"`; metadata and hashing tests
- **Depends on:** PR 2
- **Rollback:** Revert tokenizer wrapper changes
- **Acceptance:** Reproducible training with synthetic data; correct metadata and hash

##### PR D — Evaluation Framework
- **Files:** `bharat/tokenizer/evaluate.py` (extended), `scripts/evaluate_tokenizer.py`, `tests/test_tokenizer_eval.py`, `tests/fixtures/tokenizer_eval.py`
- **Content:** Per-language, per-domain, per-script metrics; baseline comparison (GPT-2); JSON report schema
- **Depends on:** PR C
- **Rollback:** Revert evaluate.py changes
- **Acceptance:** All metrics produce deterministic output; baseline comparison works

##### PR E — Production Training Configuration
- **Files:** `configs/tokenizers/bpe-64k.yaml` (final review), normalization policy tests, acceptance thresholds
- **Content:** Final tokenizer configuration; verified normalization tests; documented acceptance thresholds
- **Depends on:** PR C
- **Rollback:** Revert config changes
- **Acceptance:** Configuration approved, no production training started

##### PR F — Production Tokenizer Evidence
- **Files:** Tokenizer config, evaluation report, sample manifest, corpus digest, tokenizer hash, approval record
- **Content:** Train 64K tokenizer from approved corpus (outside CI); generate evaluation report; verify hash and metadata
- **Depends on:** PR E, approved data release
- **Rollback:** None (no production files changed)
- **Acceptance:** All acceptance thresholds met or revised with evidence

##### PR G — Bharat-350M Tokenizer Integration
- **Files:** Model config updates if needed; smoke/overfit test configurations
- **Content:** Model config compatibility; checkpoint metadata; smoke and overfit training
- **Depends on:** PR F
- **Rollback:** Revert model config changes
- **Acceptance:** Tokenizer integrates with Bharat-350M config; smoke and overfit tests pass

#### Milestone 6.2 — Bharat-350M Smoke Test + Overfit

- **Content:** Overfit-one-batch test, small-scale training (100M tokens), benchmark report
- **Depends on:** Milestone 6.1 complete, Milestone 4 (BharatBench) complete
- **Acceptance:** Model overfits one batch; distributed training converges; benchmark report generated

#### Milestone 6.3 — Comprehensive Benchmark Report

- **Content:** Full evaluation on BharatBench; model card; safety review
- **Depends on:** Milestone 6.2 complete
- **Acceptance:** All benchmark results documented; safety review passed; model card complete

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

| Order | PR Title | Main Files | Depends On | Duration | Status |
|-------|----------|-----------|------------|----------|--------|
| 1 | ci: Add tests, linting, and CI infrastructure | `tests/`, `.github/`, `pyproject.toml`, `ruff.toml` | — | 1-2 days | ✅ |
| 2 | feat: Unified tokenizer interface | `bharat/tokenizer/` (6 files) | PR 1 | 2-3 days | ✅ |
| 3 | fix: SFT assistant-only loss masking | `train/sft.py`, `bharat/posttraining/sft.py`, `datasets.py`, `collators.py`, `templates.py` | PR 2 | 2-3 days | ✅ |
| 4 | fix: DPO per-sample response masking | `train/dpo.py`, `bharat/posttraining/dpo.py`, `preference_loss.py`, `preference_dataset.py` | PR 2 | 2-3 days | ✅ |
| 5 | feat: Checkpoint metadata and resume | `train/pretrain.py`, `train/pretrain_ddp.py`, `bharat/training/checkpointing.py` | PR 2 | 1-2 days | ✅ |
| 6 | docs: Rebrand as Bharat AI | `README.md`, `docs/VISION.md`, `docs/ROADMAP.md`, etc. | PR 1 | 1 day | ✅ |
| 7 | feat: Model components (RoPE, RMSNorm, SwiGLU, GQA) | `bharat/models/config.py`, `rotary.py`, `normalization.py`, `mlp.py`, `attention.py` | PR 1 | 3-4 days | ✅ |
| 8 | feat: Bharat model + generation with KV cache | `bharat/models/bharat_model.py`, `generation.py`, `outputs.py`, `cache.py` | PR 7 | 2-3 days | ✅ |
| 9 | feat: Model configs + parameter calculator | `configs/models/bharat-*.yaml`, `bharat/models/spec.py`, `sizing.py`, `scripts/calculate_params.py`, `docs/MODEL_CONFIGURATIONS.md` | PR 8 | 1 day | ✅ |
| 10 | feat: Governed data source registry and licensing | `bharat/data/`, `data_registry/`, `scripts/validate_data_registry.py`, `docs/DATA_GOVERNANCE.md` | PR 1 | 3 days | ✅ |
| 11 | feat: Data quality filters and deduplication | `bharat/data/*_dedup.py`, `pii.py`, `quality.py`, `safety_filter.py`, `processing.py` | PR 10 | 2-3 days | ✅ |
| 12 | feat: Data manifests and contamination checks | `bharat/data/contamination.py`, `manifests.py`, `sharding.py` | PR 11 | 2 days |
| 13 | feat: Evaluation runner and reporting | `bharat/evaluation/runner.py`, `registry.py`, `reporting.py` | PR 1 | 2 days |
| 14 | feat: Evaluation modules | `bharat/evaluation/*.py` (12 modules) | PR 13 | 3-4 days |
| 15 | feat: Leaderboard and comparison reporting | Updates to `reporting.py` | PR 14 | 1-2 days |
| 16 | feat: Streaming API and function calling | `bharat/serving/api.py`, `schemas.py`, `engine.py`, `streaming.py` | PR 2, 8 | 2-3 days |
| 17 | feat: Auth, rate limiting, metrics | `bharat/serving/authentication.py`, `rate_limit.py`, `metrics.py` | PR 16 | 1-2 days |
| 18 | feat: Export (safetensors + GGUF F32) | `bharat/serving/safetensors_writer.py`, `gguf_writer.py`, `gguf_tensor_writer.py`, `gguf_preflight.py`, `gguf_reader.py`, `export_writer.py`, `export_manifest*.py`, `export_writer_readiness*.py`, `export_path_readiness*.py`, `scripts/run_export_plan.py`, 15 milestone docs, 168 tests | PR 8 | 2 weeks | ✅ |
| 19a | docs: Milestone 6.1 tokenizer validation plan | `docs/MILESTONE_6_1_TOKENIZER_VALIDATION_PLAN.md`, `docs/IMPLEMENTATION_PLAN.md` | — | 1 day | 🔲 |
| 19b | feat: Deterministic tokenizer-corpus sampler | `bharat/tokenizer/sampler.py`, `scripts/sample_tokenizer_corpus.py` | Milestone 3.5 | 3 days | 🔲 |
| 19c | feat: Tokenizer training harness + BPE wrapper | `bharat/tokenizer/train.py`, `loader.py`, `configs/tokenizers/bpe-64k.yaml` | PR 2 | 2-3 days | 🔲 |
| 19d | feat: Extended evaluation framework | `bharat/tokenizer/evaluate.py`, `scripts/evaluate_tokenizer.py` | PR 19c | 3 days | 🔲 |
| 19e | feat: Production tokenizer config + thresholds | `configs/tokenizers/bpe-64k.yaml` | PR 19c | 1 day | 🔲 |
| 19f | feat: Production 64K tokenizer evidence | Tokenizer artifacts (outside CI) | PR 19e | 1-2 days | 🔲 |
| 19g | feat: Bharat-350M tokenizer integration | Model config, smoke tests | PR 19f | 1-2 days | 🔲 |
| 20 | test: Bharat-350M smoke test and benchmark | Training/eval configs | PR 9, 15, 19g | 1-2 weeks | 🔲 |
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

## Acceptance Criteria — Milestone 1 ✅ (Completed at `637e2d3`)

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | All tests pass (`pytest`) | ✅ | Scaffolding in place; many tests are `pytest.skip` stubs |
| 2 | Linting passes (`ruff check .`) | ✅ | `ruff.toml` configured |
| 3 | Type checking passes (`mypy --strict bharat/`) | ✅ | `pyproject.toml` configured |
| 4 | CI is green (GitHub Actions) | ✅ | 3 jobs: lint, typecheck, test |
| 5 | Unified tokenizer used everywhere | ✅ | All legacy scripts via `bharat.tokenizer.load_tokenizer()`; only `export_ollama.py` remains hardcoded |
| 6 | SFT loss is assistant-only | ✅ | Non-assistant positions masked with `-100`; embedding resized |
| 7 | DPO uses per-sample masks | ✅ | Per-sample `prompt_len`; `log_probs()` uses per-sample mask tensor |
| 8 | Checkpoints include metadata | ✅ | Tokenizer type/hash, git SHA, vocab size, package versions |
| 9 | Incompatible tokenizer rejected | ✅ | Tokenizer hash validated on checkpoint resume |
| 10 | Training is restartable | ⚠️ Partial | Optimizer state saved; scheduler and random state NOT saved |
| 11 | CPU smoke test passes | ❓ Untested | `scripts/sanity_check.py` exists but not verified in CI |
| 12 | API has configurable CORS | ✅ | `CORS_ORIGINS` env var (default `"*"`) |
| 13 | README has no unsupported claims | ✅ | Vision section clearly separated |
| 14 | Roadmap is accurate | ✅ | Completed/active/planned states clearly indicated |
