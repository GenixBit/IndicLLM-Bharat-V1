# AGENTS.md — Session Log

## Session: Milestone 3.2.2 — Pipeline Correctness Final Gate

**Date:** 2026-07-20
**Branch:** `fix/milestone-3-processing-final-gate`
**HEAD:** `c5f074b` — "fix: pipeline correctness final gate"
**Upstream:** PR #6 (https://github.com/GenixBit/IndicLLM-Bharat-V1/pull/6)
**CI:** All 6 jobs green ✅

---

### Action Log

| Time | Action |
|------|--------|
| Start | Loaded repository. Branch `main` at `a239c2b` (Milestone 3.2.1 merged). |
| Planned | Designed 3 fixes: char n-gram fallback, pipeline reorder, CI verification. |
| Fixed | `bharat/data/fuzzy_dedup.py` — added char n-gram fallback when word count < n_gram_size. |
| Fixed | `bharat/data/processing.py` — reordered pipeline: quality/PII/safety checks before dedup insertion. |
| Tested | Added 5 new tests (3 for char n-gram, 2 for dedup pollution). |
| Verified | 826 tests pass, ruff/mypy clean. |
| Released | Branch `fix/milestone-3-processing-final-gate` pushed, PR #6 created. |
| Verified | All 6 CI jobs (format, lint, typecheck, diff-check, test 3.11, test 3.12) green. |

---

### Milestone 3.2.2 Changes

| File | Change |
|------|--------|
| `bharat/data/fuzzy_dedup.py` | Char n-gram fallback for whitespace-less Indic text (Tamil, Devanagari, etc.) |
| `bharat/data/processing.py` | Quality/PII/safety gates before dedup; only accepted records enter dedup |
| `tests/data/test_filters.py` | 5 new tests covering both fixes |

### Test Stats

- **Total:** 826 passed, 7 skipped, 6 deselected
- **Filter tests (`tests/data/test_filters.py`):** 116 tests (was 111)

### Still Open / Not Yet Started

- Milestone 3.3: Data manifests, sharding, source registry integration, contamination checking
- Milestone 2: Modern model architecture (RoPE, RMSNorm, SwiGLU, GQA, FlashAttention)
- Milestone 4: BharatBench evaluation framework
- Milestone 5: Production serving (streaming, auth, metrics)
- Milestone 6-7: Bharat-350M and Bharat-1B training
- Docker/Kubernetes deployment configs
