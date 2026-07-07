# Bharat AI Data Source Registry

This directory contains the versioned, governed data-source registry for
the Bharat AI project.  Sources must be individually researched and
reviewed before being added.

## Directory structure

```
data_registry/
├── README.md              ← this file
├── license_policy.yaml    ← licence allowlist / default-deny policy
├── sources/               ← individual source YAML files
└── examples/              ← templates (ignored by the loader)
```

## Important

- **No dataset is automatically legally approved.**
- **No data has been downloaded or processed.**
- The registry is an engineering control, not legal advice.
- Quality filtering, deduplication and sharding remain future work.
- The legacy `data/` pipelines remain unchanged.

## Validation

```bash
python scripts/validate_data_registry.py
```
