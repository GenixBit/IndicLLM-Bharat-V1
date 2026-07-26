# Milestone 5.3: GGUF preflight CLI integration

This increment integrates the existing deterministic GGUF metadata preflight validator into the local export-plan CLI.

## Included

- `--gguf-metadata-path` accepts a local JSON metadata file.
- The option is valid only with `--format gguf`.
- Remote metadata paths are rejected before file access.
- Checkpoint inventory is built automatically when GGUF preflight is requested.
- Successful validation is emitted as deterministic `gguf_preflight` JSON.
- Existing CLI output remains unchanged when the option is omitted.
- Focused subprocess tests use tiny local fixtures only.

## Safety boundary

This increment does not:

- train or fine-tune a model;
- download models, datasets, or benchmarks;
- call external APIs or scrape remote sources;
- upload files or workflow artifacts;
- parse binary GGUF payloads;
- load tensor payloads;
- convert checkpoints; or
- serialize model weights.

## Milestone status

Milestone 5.3 is now complete. Both local safetensors and GGUF F32 writers are implemented, tested, and integrated into the export CLI. See [MILESTONE_5_3_EXPORT_CLOSURE.md](MILESTONE_5_3_EXPORT_CLOSURE.md) for details.
