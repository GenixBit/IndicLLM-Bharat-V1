# Milestone 4.5 — BharatBench Checkpoint Leaderboard

**Status:** Implemented

## Objective

Add a leaderboard module for cross-checkpoint comparison. Aggregate
BharatBench evaluation reports into ranked tables with JSON and
Markdown export.

## Implemented in this PR

- `LeaderboardEntry` — frozen dataclass with `checkpoint_name`,
  `benchmark_id`, `category`, `metric_values`, `aggregate_score`,
  and `metadata`
- `Leaderboard` — registry with `add_entry`, `rank` (sorted by
  aggregate score descending with stable tie-breaking by checkpoint
  name), `to_json`, and `to_markdown`
- `load_report()` — reads a BharatBench report JSON file and
  constructs a `LeaderboardEntry`
- `load_leaderboard()` — scans a directory of report JSON files and
  builds a `Leaderboard`
- `scripts/build_leaderboard.py` — CLI with `--reports-dir`,
  `--output`, `--benchmark-id`, `--category`, `--format` (json/markdown)
- 5 synthetic report fixtures under `eval_fixtures/leaderboard/`
  (bharat-350m, bharat-1b, gpt2 on language_qa and reasoning_cls)
- 25 tests covering entry validation, ranking, tie-breaking,
  filtering, JSON/Markdown export, report loading, and fixture loading

## CLI Usage

```bash
# Build full leaderboard as JSON (stdout)
python scripts/build_leaderboard.py --reports-dir eval_fixtures/leaderboard/

# Build leaderboard filtered to a benchmark, output as Markdown
python scripts/build_leaderboard.py \
  --reports-dir eval_fixtures/leaderboard/ \
  --benchmark-id language_qa \
  --format markdown \
  --output leaderboard.md

# Machine-readable JSON output
python scripts/build_leaderboard.py \
  --reports-dir eval_fixtures/leaderboard/ --json
```

## Report Fixture Format

Each report JSON file in `--reports-dir` must contain at minimum:

```json
{
  "checkpoint_name": "bharat-350m",
  "benchmark_id": "language_qa",
  "aggregate_scores": {
    "overall_exact_match": 1.0,
    "overall_token_f1": 1.0
  }
}
```

Optional fields: `category`, `metadata`. The `aggregate_score` for
ranking is computed as the mean of all values in `aggregate_scores`.

## Ranking Rules

1. Entries are sorted by `aggregate_score` descending.
2. Ties are broken by `checkpoint_name` alphabetically (stable).
3. Filtering by `benchmark_id` and/or `category` is supported.

## API

```python
from bharat.eval import Leaderboard, LeaderboardEntry, load_leaderboard

lb = load_leaderboard(Path("eval_fixtures/leaderboard"))
for entry in lb.rank(benchmark_id="language_qa"):
    print(entry.checkpoint_name, entry.aggregate_score)

print(lb.to_markdown(category="reasoning"))
```

## Offline Guarantee

All report fixtures are synthetic and checked into the repository.
No external APIs, downloads, or network calls.
