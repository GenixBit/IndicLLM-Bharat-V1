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

---

## Session: Milestone 6.1 — Candidate Evidence Integrity Hardening (PR #69 iteration)

**Date:** 2026-07-29
**Branch:** `fix/milestone-6-1-candidate-evidence-integrity`
**HEAD:** `06c454e` → *(pending commit)*
**Upstream:** PR #69 open as draft.

### Action Log

| Time | Action |
|------|--------|
| Start | Loaded branch `fix/milestone-6-1-candidate-evidence-integrity` at `06c454e`. |
| Step 1 | Rewrote `production_evidence_builder.py`: removed global ownership state (`_OWNERSHIP_MARKER`, `_OWNED_FILES`, `_register_owned`, `_cleanup_owned`, `_cleanup_stale_temps`). |
| Step 2 | Rewrote `_publish_exclusive` with local `created` flag and self-cleaning on failure. |
| Step 3 | Added `_write_temp` with bounded retry (`_MAX_TEMP_RETRIES=16`) and `secrets.token_hex` for temporary-file collision safety. |
| Step 4 | Added success-path temp cleanup in `write_candidate_manifest`. |
| Step 5 | Updated `_check_output_path` to reject all pre-existing filesystem objects (files, dirs, symlinks). |
| Step 6 | Added canonical-equivalence record (rec-4, `canonical_equivalent`) to `build_input_jsonl` in fixtures. |
| Step 7 | Updated `build_production_thresholds` with `min_canonical_evaluated_count: 1` and `min_records_per_required_language: 1`. |
| Step 8 | Removed unused `tokenizer_fp` argument from `build_acceptance_decision` signature and all call sites. |
| Step 9 | Removed file-level `# ruff: noqa: F811` suppression; re-exported `evidence_fixtures` via `tests/tokenizer/conftest.py`. |
| Step 10 | Removed unused `BPETokenizer` import and `tokenizer_fp` variable from CLI test file. |
| Step 11 | Replaced 9 old forged-report tests with 9 metric-specific forgery tests (micro_fertility, unknown_token_rate, required_pass_rate, canonical_evaluated_count, canonical_pass_rate, byte_coverage, fragmentation, per-language metrics, failed_records). |
| Step 12 | Added 10 failure-injection tests using real filesystem scenarios (output dir/symlink, tokenizer dir/broken-symlink/symlink-outside-root, input symlink outside root, evidence root is file/missing). |
| Step 13 | Updated `test_per_language_count_derivation` for canonical record (hi: 1→2). |
| Step 14 | Verified: 58/58 builder tests ✓, 6/6 CLI tests ✓, 63/63 acceptance tests ✓, 343/343 tokenizer suite ✓, ruff check ✓, ruff format ✓, mypy ✓. |

### Changes (round 3)

| File | Change |
|------|--------|
| `bharat/tokenizer/production_evidence_builder.py` | Removed global ownership state; rewrote `_publish_exclusive` with local self-cleaning; added `_write_temp` with bounded retry; updated `_check_output_path` to reject all filesystem objects; added success-path temp cleanup |
| `tests/tokenizer/evidence_fixtures.py` | Added canonical-equivalence record (rec-4); updated thresholds with `min_canonical_evaluated_count` and `min_records_per_required_language`; removed `tokenizer_fp` from `build_acceptance_decision` |
| `tests/tokenizer/conftest.py` | **NEW** — re-exports `evidence_fixtures` fixture for pytest discovery |
| `tests/tokenizer/test_production_evidence_builder.py` | Removed `# ruff: noqa: F811` suppression; replaced 9 forged tests with metric-specific ones; added 10 failure-injection tests; updated `build_acceptance_decision` calls; fixed hi count (1→2) |
| `tests/scripts/test_build_production_tokenizer_evidence.py` | Removed unused `tokenizer_fp` arg from `_build_report` and `_build_decision`; removed unused `BPETokenizer` import and `tokenizer_fp` variable |

### Test Stats (final)

- **Builder tests:** 58 passed
- **CLI tests:** 6 passed
- **Acceptance tests:** 63 passed
- **Tokenizer suite:** 343 passed
- **Total relevant:** 409 passed
