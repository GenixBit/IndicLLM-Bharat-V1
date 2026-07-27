# Tokenizer Evaluation Framework

**Status: In review**

## Evaluation scope

The tokenizer evaluation framework measures deterministic, offline metrics for
any `BharatTokenizer`-compatible implementation. It evaluates tokenization
quality on synthetic or approved local evaluation fixtures only.

### What it evaluates

- Per-record tokenization metrics (fertility, round-trip fidelity, unknown
  tokens, byte coverage);
- Aggregate statistics grouped by language, script, domain and category;
- Fragmentation patterns (words, numbers, punctuation, URLs, code identifiers,
  emoji/ZWJ sequences);
- Round-trip correctness (exact and NFC-normalized);
- Canonical-equivalence verification;
- Multi-tokenizer comparison (cross-tokenizer wins, ties, losses).

### What it does NOT do

- Train the production 64K tokenizer;
- Download datasets, models or tokenizer artifacts;
- Access Hugging Face or external APIs;
- Execute model training or inference;
- Resize model embeddings;
- Make unsupported production-quality claims;
- Access the network in any form.

## Metric formulas

### Character and byte definitions

- **Unicode character count** = number of Unicode code points in the text
  (equivalent to `len(text)` in Python).
- **UTF-8 byte count** = `len(text.encode("utf-8"))`.
- Whitespace, combining marks, zero-width joiners and variation selectors are
  counted as characters.
- Control characters (U+0000–U+001F, U+007F–U+009F) are counted as characters
  when present.

### Token fertility

| Metric | Formula |
|---|---|
| Tokens per character (fertility) | `token_count / char_count` |
| Tokens per code point | `token_count / codepoint_count` |
| Tokens per UTF-8 byte | `token_count / utf8_byte_count` |
| Characters per token | `char_count / token_count` |
| Bytes per token | `utf8_byte_count / token_count` |

When `char_count` or `token_count` is zero, the corresponding ratio is 0.0.
Division by zero is never silent.

- **Micro-average fertility**: total tokens across all records divided by total
  characters.
- **Macro-average fertility**: mean of per-record fertility values.
- **Median fertility**: median of per-record fertility values.

### Aggregate statistics

Each group (language, script, domain, category) reports:

- `record_count`
- `char_count`
- `byte_count`
- `token_count`
- `micro_fertility`
- `macro_fertility`
- `min_fertility`
- `max_fertility`
- `median_fertility`

### Round-trip fidelity

- **Exact round trip**: `decode(encode(text)) == text` for input text.
- **NFC round trip**: `decode(encode(NFC(text))) == NFC(text)`.
- **Canonical equivalence**: when the record provides a `canonical_equivalent`
  field, `decode(encode(canonical_equivalent)) == NFC(text)`.

### Unknown-token interpretation

- **Unknown token count**: number of token IDs equal to `unk_token_id`.
- **Unknown token rate**: `unknown_token_count / token_count`.
- A record containing one or more unknown tokens is recorded as affected.
- Valid Unicode text that produces unknown tokens indicates a vocabulary gap.

### Byte-coverage interpretation

- **Byte-alphabet completeness**: whether all 256 byte values (0x00–0xFF) are
  reachable through the tokenizer.
- Missing byte values are reported individually.
- For `BharatBPETokenizer`, all 256 bytes are always reachable.

## Fragmentation methodology

The framework segments decoded text using documented deterministic regular
expressions. These are **not** linguistically complete parsers.

| Category | Segmentation rule |
|---|---|
| Words | `\w+` (Unicode-aware) |
| Numbers | `\d+(?:[.,]\d+)*` |
| Punctuation | One or more ASCII punctuation characters |
| URLs | `https?://...` or `www....` |
| Emails | Basic email pattern |
| Hashtags | `#\w+` |
| Code identifiers | Snake_case, camelCase, single letters |
| CamelCase | Capital/lowercase segments |
| Snake_case | `[a-z]+(?:_[a-z]+)*` |
| Mixed Indic+Latin | Contains both Latin and non-Latin characters |
| Emoji/ZWJ | Emoticons, symbols, ZWJ sequences |

For each category, the framework reports:

- **Item count**: number of matched items (capped at 1000 per category).
- **Total tokens**: sum of whitespace-split segments across items.
- **Average tokens per item**.
- **Percentage represented by one token**: items whose whitespace-split length
  ≤ 1.
- **Percentage represented by more than four tokens**: items whose
  whitespace-split length ≥ 4.
- **Maximum token count**: longest item by whitespace-split length.

## Comparison methodology

When two or more tokenizers are provided:

- **Win**: lower token count when both tokenizers pass the required round-trip
  contract.
- **Tie**: equal token count when both pass round-trip.
- **Loss**: higher token count when both pass round-trip.
- Records where either tokenizer fails round-trip are excluded from wins/ties.
- Per-language fertility differences are calculated as `fertility_A -
  fertility_B`.
- Absolute and relative fertility differences are reported overall.
- Lower token count is one signal, not the sole measure of quality.

## Canonical report schema

```json
{
  "schema_version": "eval-v1",
  "evaluator_version": "1.0.0",
  "input_dataset_sha256": "<sha256>",
  "tokenizer_names": ["..."],
  "tokenizer_fingerprints": {"name": "..."},
  "aggregate": {
    "tokenizer_name": {
      "record_count": 0,
      "char_count": 0,
      "byte_count": 0,
      "token_count": 0,
      "micro_fertility": 0.0,
      "macro_fertility": 0.0,
      "min_fertility": 0.0,
      "max_fertility": 0.0,
      "median_fertility": 0.0,
      "unknown_token_count": 0,
      "records_with_unknown": 0,
      "unknown_token_rate": 0.0,
      "special_token_count": 0,
      "byte_token_count": 0,
      "merged_token_count": 0
    }
  },
  "per_language": {"name": {"lang": {...}}},
  "per_script": {"name": {"script": {...}}},
  "per_domain": {"name": {"domain": {...}}},
  "per_category": {"name": {"category": {...}}},
  "round_trip": {
    "name": {
      "exact_pass_count": 0,
      "exact_pass_rate": 0.0,
      "nfc_pass_count": 0,
      "nfc_pass_rate": 0.0,
      "failed_record_ids": [],
      "failure_categories": {}
    }
  },
  "fragmentation": {
    "name": {
      "words": {"item_count": 0, ...},
      "numbers": {...},
      "punctuation": {...},
      ...
    }
  },
  "comparison": [
    {
      "tokenizer_a": "...",
      "tokenizer_b": "...",
      "absolute_token_count_difference": 0,
      "relative_fertility_difference": 0.0,
      "per_language_fertility_difference": {},
      "wins_a": 0,
      "wins_b": 0,
      "ties": 0,
      "round_trip_difference": {},
      "unknown_token_difference": {}
    }
  ],
  "failed_records": [{"record_id": "...", "tokenizer": "..."}],
  "report_sha256": "<sha256>"
}
```

## Determinism guarantees

- Stable key ordering (`sort_keys=True` in JSON output).
- Deterministic list ordering (sorted by record ID, tokenizer name, language,
  etc.).
- Compact canonical JSON (`separators=(",", ":")`, `ensure_ascii=True`).
- No wall-clock timestamp in canonical bytes.
- No host paths or environment-dependent values.
- No raw evaluation text in the default report.
- Report digest (`report_sha256`) is computed over all other fields; the digest
  field is excluded from its own calculation.
- Identical inputs produce byte-identical reports.

## Privacy behavior

- The default report contains no raw evaluation text.
- Failed record IDs are reported (without text) when round-trip checks fail.
- Detailed per-record output is available only when explicitly requested via
  `--detailed-records`.
- No network access occurs during evaluation.
- No host-identifying information is included in the report.

## Local-only security boundary

- All evaluation inputs must be local file paths.
- No URLs or remote paths are accepted for tokenizer artifacts or datasets.
- No data is uploaded, downloaded or sent over the network.
- The framework can execute fully offline with no network access.

## Fixture limitations

- Language fixtures are tiny synthetic samples (1–5 records each).
- Fixtures do not represent real-world language distribution.
- Evaluation results on fixtures are not production-quality measurements.
- Fixtures cover common orthographic features but not dialectal variation.

## Known limitations

- Fragmentation uses regex-based segmentation, not linguistic parsing.
- Word boundary detection follows Unicode `\w+` which may not match all
  language-specific word boundaries.
- Fertility comparison treats lower token count as a win; does not account for
  information preservation.
- Unknown-token detection relies on `unk_token_id` being correctly exposed.
- Byte-coverage detection attempts to encode each single-byte character
  (`latin-1` decoded), which may fail for non-printable bytes depending on
  tokenizer implementation.
- Comparison excludes records where either tokenizer fails round-trip, which
  may bias results toward simpler texts.

## Next milestone decision criteria

The evaluation framework is considered stable when:

1. The report schema version is finalized (currently `eval-v1`).
2. All 40+ focused tests pass deterministically across runs.
3. The tiny BPE artifact passes round-trip checks (100% exact and NFC).
4. Byte-coverage confirms all 256 bytes are reachable.
5. Per-language aggregate metrics produce non-zero, non-identical values.
6. The framework runs successfully on the production tokenizer without errors.
7. Baseline comparison methodology is reviewed and approved.

Production 64K tokenizer training remains separately blocked until:

- The evaluation schema is stable;
- The tiny BPE artifact passes round-trip and byte-coverage checks;
- Baseline comparison methodology is reviewed;
- Corpus and compute plans are separately approved.
