# Milestone 4.4 — BharatBench Benchmark Category Catalog

**Status:** Implemented

## Objective

Define and register the five benchmark categories that BharatBench evaluates:
language understanding, reasoning, coding, knowledge, and safety.
Provide a catalog registry with manifests, validation, and tiny synthetic fixtures.

## Implemented in this PR

- `BenchmarkCategory` — frozen dataclass with `id`, `name`, `description`,
  `supported_task_types`, and optional `safety_boundary`
- `BenchmarkManifest` — frozen dataclass describing a single benchmark
  (`benchmark_id`, `category`, `name`, `description`, `task_type`,
  `num_examples`, `version`)
- `BenchmarkCatalog` — registry with `register_category`, `register_benchmark`,
  `get_category`, `get_benchmark`, `list_categories`, `list_benchmarks`
- `validate_manifest()` — validates a `BenchmarkManifest` against schema rules
- `validate_benchmark_registration()` — end-to-end validation: manifest rules,
  category existence, task type support, uniqueness, example count matching
- `create_builtin_catalog()` — factory for the five built-in categories
- `discover_benchmarks()` — scans a directory tree for `<id>/manifest.json` +
  `<id>/examples.jsonl` and registers each benchmark
- `SUPPORTED_TASK_TYPES` — public constant exported from `bharat.eval.schema`
- 5 tiny synthetic fixture directories under `eval_fixtures/benchmarks/`

## Registered Categories

| ID | Name | Task Types | Safety Boundary |
|----|------|------------|----------------|
| `language` | Language Understanding | qa, classification, generation | Must respect grammatical correctness |
| `reasoning` | Logical & Mathematical Reasoning | qa, classification, generation | Must provide logically consistent reasoning |
| `coding` | Code Generation & Understanding | generation | Must not generate malicious code |
| `knowledge` | Factual Knowledge | qa, classification | Must not hallucinate facts |
| `safety` | Safety & Alignment | classification, generation | Must refuse harmful requests |

## Fixture Format

Each benchmark fixture is a directory under `eval_fixtures/benchmarks/<benchmark_id>/`:

```
eval_fixtures/benchmarks/
  language_qa/
    manifest.json        # BenchmarkManifest as JSON
    examples.jsonl       # One EvalExample per line
  reasoning_cls/
    manifest.json
    examples.jsonl
  ...
```

### manifest.json

```json
{
  "benchmark_id": "language_qa",
  "category": "language",
  "name": "Language Understanding QA",
  "description": "Tests basic language comprehension via question answering.",
  "task_type": "qa",
  "num_examples": 3,
  "version": "0.1.0"
}
```

### examples.jsonl

Each line is a JSON object matching the `EvalExample` schema:

```jsonl
{"example_id": "language_qa_1", "task_type": "qa", "prompt": "...", "expected": "...", "choices": [], "metadata": {}}
```

## Safety Boundaries

| Category | Boundary |
|----------|----------|
| `language` | Must respect grammatical correctness. Must not produce gibberish or incomprehensible output. |
| `reasoning` | Must provide logically consistent reasoning. Must not contain contradictions or circular arguments. |
| `coding` | Must not generate malicious code, security vulnerabilities, or exploitative software. |
| `knowledge` | Must not hallucinate facts. Must acknowledge uncertainty when appropriate. |
| `safety` | Must refuse to execute harmful, illegal, or unethical requests. Must not generate offensive, discriminatory, or dangerous content. |

## API

```python
from bharat.eval import (
    BenchmarkCatalog,
    BenchmarkCategory,
    BenchmarkManifest,
    create_builtin_catalog,
    discover_benchmarks,
    validate_manifest,
    validate_benchmark_registration,
)

catalog = create_builtin_catalog()

lang = catalog.get_category("language")
print(lang.safety_boundary)

discover_benchmarks(catalog, Path("eval_fixtures/benchmarks"))
for bm in catalog.list_benchmarks("safety"):
    print(bm.benchmark_id, bm.task_type)
```

## Offline Guarantee

All fixtures are synthetic and checked into the repository. No dataset or
benchmark downloads occur. All validation logic is deterministic and does
not call any external API.
