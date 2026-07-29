# AGENTS.md — Session Log

## Session: Milestone 6.1 — Tokenizer Acceptance Gate Hardening (PR #63 review)

**Date:** 2026-07-28
**Branch:** `feat/milestone-6-1-tokenizer-acceptance-gate`
**HEAD:** `add8242`
**Upstream:** Pushed to origin. PR #63 updated.

---

### Action Log

| Time | Action |
|------|--------|
| Start | Loaded branch `feat/milestone-6-1-tokenizer-acceptance-gate` at `96c5860`. |
| Step 1 | Read all source and test files for acceptance, evaluation, and CLI modules. |
| Step 2 | Added `canonical_evaluated: bool` to `RecordMetrics` in evaluation.py. |
| Step 3 | Added `canonical_evaluated_count` to `RoundTripSummary` and `_build_report`. |
| Step 4 | Fixed `canonical_pass_rate` denominator to use `canonical_evaluated_count`. |
| Step 5 | Updated validation in `_validate_round_trip_values` and `_validate_cross_field_consistency`. |
| Step 6 | Made `ThresholdConfiguration` the authoritative API — removed separate `thresholds` param. |
| Step 7 | Added `min_canonical_evaluated_count` threshold field with validation. |
| Step 8 | Added canonical_evaluated_count check in `_add_canonical_pass_checks`. |
| Step 9 | Hardened config validation: non-string notes, duplicate/empty language names, production scope. |
| Step 10 | Fixed dry-run exit: returns 2 on threshold failure instead of always 0. |
| Step 11 | Added digest recomputation verification in CLI publication. |
| Step 12 | Updated all tests for new API and added 16 new tests. |
| Step 13 | Verified: 75/75 tests pass, ruff format ✓, ruff check ✓. |
| Step 14 | Committed add8242, pushed, updated PR #63 description. |
| Step 15 | Identified CI mypy error: `str | None` to `encode()`. |
| Step 16 | Fixed via `if record.canonical_equivalent is not None` (type narrowing). |
| Step 17 | Updated config to `min_canonical_evaluated_count: 1`. |
| Step 18 | Made `canonical_pass_rate` null when `canonical_evaluated_count` is 0. |
| Step 19 | Used exclusive creation (`xb`) for temporary publication file. |
| Step 20 | Added 4 new tests (committed config, zero evidence, temp collision). |
| Step 21 | Verified: mypy ✓, ruff ✓, 79 tests ✓, CI all green (run #352). |
| Step 22 | Committed d79a528, pushed, updated PR as ready for review. |

---

### Changes (round 2)

| File | Change |
|------|--------|
| `bharat/tokenizer/evaluation.py` | Restored type-narrowing `if record.canonical_equivalent is not None`, `canonical_pass_rate` is `float | None`, null when 0 evaluated |
| `configs/tokenizers/bpe-64k-acceptance.json` | Added `min_canonical_evaluated_count: 1` |
| `scripts/check_tokenizer_acceptance.py` | Exclusive `xb` for tmp file, updated docstring |
| `tests/tokenizer/test_acceptance.py` | Added 4 tests (committed config, zero evidence pass/fail) |
| `tests/scripts/test_check_tokenizer_acceptance.py` | Added temp name collision test |

### Test Stats (final)

- **Acceptance tests:** 62 passed
- **CLI tests:** 17 passed
- **Total relevant:** 79 passed
