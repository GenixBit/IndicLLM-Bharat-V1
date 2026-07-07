# Bharat AI Data Registry

This directory contains the governed data-source registry for the Bharat AI project.

## Policy

- `license_policy.yaml` — defines the project's data licensing policy.
- `default_decision` is permanently set to `deny` and cannot be changed through a policy file.
- Unknown or missing licence identifiers always resolve to `DENY`.
- Approval is an engineering gate, not legal advice.

## Sources

- `sources/` — individual YAML files, one per source version.
- Each file represents a single immutable `(source_id, version)` record.
- Production registry may validly remain empty.

## Key requirements

- **Immutable revisions**: HuggingFace sources must use a 40-character lowercase hex commit SHA.
- **Integrity**: Source revision must match `integrity.revision` when integrity is present.
- **Licensing**: An `ALLOW` record requires complete evidence (URL, verifier, date, all flags).
- **Commercial use** and **model training** must be explicitly allowed for approval.
- **Versions**: PEP 440-compatible (e.g., `1.0.0`, `2.0.0`, `10.0.0`, `1.0.0-alpha`).
- **Supersession**: Format is `source_id@version` (e.g., `fineweb_edu@1.0.0`).
- **Digest**: Covers all fields including notes, plus policy representation. Policy changes alter the digest.

## Registry digest

- `registry_digest` — SHA-256 over canonical JSON of all sources plus the policy.
- `policy_digest` — SHA-256 over canonical JSON of the policy alone.
- Digest changes when: notes change, license identifiers change, integrity changes, status changes, supersession changes, or policy changes.
- Digest is deterministic (ordering-independent).

## CLI

```bash
python scripts/validate_data_registry.py
python scripts/validate_data_registry.py --json
python scripts/validate_data_registry.py --strict
```

## No dataset has been downloaded or processed.

This registry is an empty governance structure. No data has been downloaded, filtered, deduplicated, or trained on.
