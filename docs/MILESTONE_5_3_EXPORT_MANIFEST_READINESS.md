# Milestone 5.3: export manifest readiness validation

This increment adds a deterministic local safety gate for export manifest targets.

## Included

- Requires the manifest parent to exist and be a directory.
- Rejects existing manifest files to prevent overwrite.
- Rejects a manifest path equal to the export output path.
- Rejects manifest paths inside the checkpoint directory.
- Resolves relative paths, `..`, and symlinks before validation.
- Returns stable JSON-serializable readiness metadata.
- Uses focused offline tests with tiny local fixtures.

## Safety boundary

This increment does not:

- train or fine-tune a model;
- download models, datasets, or benchmarks;
- call external APIs or scrape remote sources;
- upload files or workflow artifacts;
- parse tensor or GGUF payloads;
- convert checkpoints;
- create a manifest or export file; or
- serialize model weights.

The validator performs local filesystem metadata checks only and does not modify files.

## Milestone status

Milestone 5.3 remains incomplete. The repository still uses dry-run writers and does not contain real safetensors or GGUF model-weight serialization.
