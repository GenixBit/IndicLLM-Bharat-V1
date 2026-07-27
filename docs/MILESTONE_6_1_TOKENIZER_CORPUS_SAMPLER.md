# Milestone 6.1 — Deterministic Tokenizer-Corpus Sampler

**Status:** In review
**PR:** #59

## Objective

Deterministic sampler that consumes only approved local dataset releases, produces a deterministic UTF-8 corpus and provenance manifest, without training a tokenizer or downloading data.

## Implemented

### Core Module — `bharat/tokenizer/sampler.py`

- **`SamplerConfig`** — frozen dataclass with version, seed, caps, field names, output paths, dedup flag
- **`ProvenanceRecord`** — frozen dataclass with release/approval/manifest/shard provenance per record
- **`CorpusManifest`** — frozen dataclass with full sampling metadata, per-source/language/domain tracking, and `compute_digest()` over canonical JSON
- **`sample_tokenizer_corpus()`** — main entry point consuming release roots, manifest paths, approval paths and config

### Governance Validation Chain

Full validation before any record processing:
1. Release metadata, audit report, manifest, and approval loaded from paths
2. Cross-validation: audit↔release, manifest↔release, approval↔manifest digest checks
3. Approval status must be `"approved"`; all four review flags (`license_reviewed`, `pii_reviewed`, `contamination_reviewed`, `safety_reviewed`) must be `True`
4. Shard SHA-256 verification against manifest
5. Symlink escape protection via `Path.resolve().relative_to()`

### Path Security

- Rejects remote/network paths: `http://`, `https://`, `ftp://`, `s3://`, `gs://`, `hf://`, `//`
- Symlink escape detection: resolved shard path must be relative to release root

### Record Selection

- Only records with `"accepted": true` are considered
- Empty/whitespace-only text records are skipped
- Lone surrogates (U+D800–U+DFFF) are rejected
- Malformed UTF-8 in shard files is rejected
- Embedded newlines, tabs, and spaces are preserved

### Ranking and Dedup

- Selection key: SHA-256 of canonical JSON `[version, seed, release_id, source_id, shard_digest, record_index, content_sha256]`
- Sort by `(selection_key, record_id)` for deterministic ordering
- Exact dedup: first-by-selection-key wins (SHA-256 content hash); `--no-dedup` flag

### Cap Application (Strict Precedence)

1. Per-source record cap → per-source byte cap
2. Per-language record cap → per-language byte cap
3. Global record cap → global byte cap

Never partially writes a record. Order of cap application is deterministic and tracked per counter.

### Output Format

- Corpus: deterministic UTF-8 JSONL `{"text":"..."}` with canonical compact JSON, one record per line, LF terminated
- Provenance records in manifest contain digests but no raw text

### Atomic Publication

- Write to `.tmp.{pid}.{name}`, flush+stat+verify SHA-256, `os.replace` to final
- Rollback removes corpus if manifest write fails
- Reject existing output files
- Dry-run by default (`--execute` to write); dry-run returns exact projection (same digest and bytes as execute), creates no files

### Manifest Schema

| Field | Description |
|-------|-------------|
| `schema_version` | `"1"` |
| `sampler_config` | Full `SamplerConfig` dict |
| `releases` | Per-release metadata (digests, counts) |
| `total_candidates` | Records before dedup/caps |
| `total_selected` | Records after all filters |
| `exact_dedup_removed` | Count of duplicates removed |
| `per_source_cap_removed` | Count of source-capped records |
| `per_language_cap_removed` | Count of language-capped records |
| `global_cap_removed` | Count of globally-capped records |
| `total_corpus_bytes` | Total UTF-8 bytes in corpus |
| `per_source_records/bytes` | Per-source breakdown |
| `per_language_records/bytes` | Per-language breakdown |
| `per_domain_records/bytes` | Per-domain breakdown |
| `corpus_sha256` | SHA-256 of corpus file bytes |
| `records` | Provenance records (no raw text) |
| `manifest_sha256` | SHA-256 of manifest JSON (excluding own field) |

### CLI — `scripts/sample_tokenizer_corpus.py`

- Repeatable `--release-root`, `--manifest-path`, `--approval-path` (count must match)
- `--version`, `--seed`, `--output-corpus`, `--output-manifest`
- Cap parsers: `--max-records-per-source`, `--max-bytes-per-source`, `--max-records-per-language`, `--max-bytes-per-language`
- `--max-total-records`, `--max-total-bytes`
- `--text-field`, `--language-field`, `--domain-field`
- `--no-dedup` disables exact content deduplication
- Dry-run by default; `--execute` to write
- JSON output with status, counts, and digests

### Tests — 19 CLI + 66 unit = 85 tests

- `tests/tokenizer/test_sampler.py` — 66 unit tests covering config validation, governance chain, dedup, caps, path security, determinism, dry-run/projection, output-path safety, release-total validation, atomic publish, fault injection
- `tests/scripts/test_sample_tokenizer_corpus.py` — 19 CLI integration tests

## Files Changed

| File | Change |
|------|--------|
| `bharat/tokenizer/sampler.py` | New — core sampler logic (746 lines) |
| `scripts/sample_tokenizer_corpus.py` | New — CLI entry point (202 lines) |
| `bharat/tokenizer/__init__.py` | Exported `CorpusManifest`, `ProvenanceRecord`, `SamplerConfig`, `sample_tokenizer_corpus` |
| `tests/tokenizer/test_sampler.py` | New — 34 unit tests (997 lines) |
| `tests/scripts/test_sample_tokenizer_corpus.py` | New — 29 CLI tests |

## Safety Boundary

Reads only local approved dataset releases. No network access, no data download, no tokenizer training, no model weights. All test fixtures are synthetic.
