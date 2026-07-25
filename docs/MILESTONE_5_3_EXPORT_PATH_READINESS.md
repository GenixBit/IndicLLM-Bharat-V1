# Milestone 5.3: export path readiness validation

This increment adds a deterministic local safety gate for export input and target paths.

## Included

- Resolves local output, manifest, and metadata paths before validation.
- Rejects duplicate metadata paths.
- Requires each metadata path to exist and be a regular file.
- Rejects metadata paths that collide with the export output path.
- Rejects metadata paths that collide with the export manifest path.
- Returns stable JSON-serializable readiness metadata with sorted metadata paths.
- Uses focused offline tests with tiny local fixtures.

## Safety boundary

This increment does not:

- train or fine-tune a model;
- download models, datasets, or benchmarks;
- call external APIs or scrape remote sources;
- upload files or workflow artifacts;
- parse tensor or GGUF payloads;
- convert checkpoints;
- create export or manifest files; or
- serialize model weights.

The validator performs local filesystem metadata checks only and does not modify files.

## Milestone status

Milestone 5.3 remains incomplete. The repository still does not contain a real safetensors or GGUF weight writer. This gate reduces overwrite risk before a later approved local writer implementation.
