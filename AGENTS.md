# AGENTS.md — Session Log

## Session: Post-Milestone 1 Fresh Audit & Docs Update

**Date:** 2025-07-06
**Branch:** `main`
**HEAD:** `637e2d3f0f496dd0f1cf6f4dfcfac03a793b97cd` — "feat: Milestone 1 — stabilisation, unified tokenizer, SFT/DPO fixes, CI, docs"
**Status:** All Milestone 1 work committed and pushed to `origin/main`.

---

### Action Log

| Time | Action |
|------|--------|
| Start | Loaded repository at `/Users/manojnandanwar/IndicLLM-Bharat-V1`. |
| Inspected | Verified git state: clean working tree, up to date with `origin/main`. |
| Inspected | Read `AGENTS.md` — file does not exist yet. |
| Inspected | Read `docs/CURRENT_STATE_AUDIT.md` (700 lines) and `docs/IMPLEMENTATION_PLAN.md` (312 lines) — both document pre-Milestone-1 state. |
| Audited | Ran comprehensive codebase audit via `task` agent — inspected all test files, tokenizer imports in legacy scripts, checkpoint metadata, SFT/DPO loss masking, API CORS config, CI config, docs. |
| Created | `AGENTS.md` — this file. |
| Updated | `docs/CURRENT_STATE_AUDIT.md` — fresh audit reflecting post-Milestone 1 state. |
| Updated | `docs/IMPLEMENTATION_PLAN.md` — Milestone 1 marked completed, PRs renumbered. |

---

### Key Findings from Fresh Audit

**Milestone 1 is fully implemented and committed at `637e2d3`:**

| Category | Status | Details |
|----------|--------|---------|
| **Tests + CI** | ✅ Complete | `tests/` with 13 test files + conftest, `.github/workflows/ci.yml` (3 jobs), `ruff.toml`, `pyproject.toml` (pytest + mypy), `.pre-commit-config.yaml`. Note: many tests are `pytest.skip` stubs awaiting actual implementation. |
| **Unified tokenizer** | ✅ Complete | `bharat/tokenizer/` (6 files): abstract base + GPT-2/SentencePiece/HF wrappers + training + evaluation + metadata. All legacy scripts import from `bharat.tokenizer.load_tokenizer()` — no hardcoded `GPT2TokenizerFast` except `export_ollama.py` (HF conversion utility). |
| **SFT loss masking** | ✅ Complete | `train/sft.py` masks non-assistant positions with `-100` in labels. Embedding resized after `add_special_tokens`. `bharat/posttraining/collators.py` also implements assistant-prefix masking. |
| **DPO per-sample masking** | ✅ Complete | `train/dpo.py` returns per-sample `prompt_len`, `log_probs` uses per-sample mask tensor (not `prompt_len[0]`). |
| **Checkpoint metadata** | ✅ Complete | Both `pretrain.py` and `pretrain_ddp.py` store tokenizer type/hash, git SHA, vocab size, package versions. Validate tokenizer on resume. |
| **API CORS** | ✅ Complete | `inference/api.py` reads `CORS_ORIGINS` env var (defaults to `"*"`), uses `TOKENIZER.eos_token_id` (not hardcoded 50256). |
| **Documentation** | ✅ Complete | All 9 planned docs exist: `README.md`, `ARCHITECTURE.md`, `VISION.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, `RELEASE_PROCESS.md`, `CURRENT_STATE_AUDIT.md`, `IMPLEMENTATION_PLAN.md`. |

**Still Open / Not Yet Started:**
- Milestone 2: Modern model architecture (RoPE, RMSNorm, SwiGLU, GQA, FlashAttention)
- Milestone 3: Data engine (dedup, PII, manifests, contamination)
- Milestone 4: BharatBench evaluation framework
- Milestone 5: Production serving (streaming, auth, metrics)
- Milestone 6-7: Bharat-350M and Bharat-1B training
- Streaming API support
- Authentication / rate limiting
- Data manifests, deduplication, PII filtering
- Contamination checking
- Safety docs / model cards
- Test implementations (most tests are stubs)

---

### Session Completion Checklist

- [x] Fresh audit of codebase state completed
- [x] `AGENTS.md` created with session log
- [x] `docs/CURRENT_STATE_AUDIT.md` updated to post-Milestone 1 state
- [x] `docs/IMPLEMENTATION_PLAN.md` updated — Milestone 1 marked completed
- [ ] Changes committed on a new branch
- [ ] Changes pushed to remote

---

### Next Steps (Suggested)

1. Implement Milestone 2 (Modern Architecture — models with RoPE/RMSNorm/SwiGLU/GQA)
2. Fill in stub tests with real test logic
3. Add streaming support to the API
4. Begin Milestone 3 (Data Engine)
5. Train Bharat-350M tokenizer and run smoke test
