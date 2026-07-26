# Milestone 5.3: export writer readiness validation

This increment adds a deterministic local safety gate for future export writers.

## Included

- Confirms the checkpoint inventory belongs to the export plan checkpoint.
- Requires a non-empty, internally consistent checkpoint inventory.
- Requires the output parent to exist and be a directory.
- Rejects an existing output path to prevent accidental overwrite.
- Rejects output paths inside the checkpoint directory.
- Returns stable, JSON-serializable readiness metadata.
- Uses focused local tests with tiny fixtures.

## CLI integration

The `--validate-writer-readiness` flag integrates the safety gate into the local export CLI:

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --validate-writer-readiness
```

### Execution ordering

1. Build the `ExportRequest` and `ExportPlan`.
2. Build the checkpoint inventory automatically.
3. Run any requested format-specific metadata preflight (safetensors or GGUF).
4. Run writer readiness validation.
5. Invoke the dry-run export writer.

### JSON output shape

When `--validate-writer-readiness` is supplied, a `writer_readiness` object is added to the CLI output:

```json
{
  "writer_readiness": {
    "checkpoint_path": "/abs/path/to/checkpoint",
    "output_path": "/abs/path/to/output.safetensors",
    "export_format": "safetensors",
    "checkpoint_file_count": 4,
    "checkpoint_total_bytes": 1048576,
    "output_parent": "/abs/path/to",
    "output_exists": false,
    "output_inside_checkpoint": false,
    "ready": true
  }
}
```

### Failure behaviour

- Exits with a non-zero status.
- Prints a deterministic error message to stderr.
- Does not invoke the export writer.
- Does not create an export manifest.
- Does not create or modify the output file.
- Does not partially emit JSON to stdout.

### Safety protections

| Protection | Behaviour |
|-----------|-----------|
| No-overwrite | Rejects output path that already exists |
| Output location | Rejects output path inside the checkpoint directory |
| Parent directory | Requires output parent to exist and be a directory |
| Path resolution | Uses `Path.resolve()` to handle `..`, relative paths, and symlinks |

### Examples

**Safetensors:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --validate-writer-readiness \
    --safetensors-metadata-path ./safetensors-metadata.json
```

**GGUF:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.gguf \
    --format gguf \
    --model-name bharat-local \
    --validate-writer-readiness \
    --gguf-metadata-path ./gguf-metadata.json
```

## Safety boundary

This increment does not:

- train or fine-tune a model;
- download models, datasets, or benchmarks;
- call external APIs or scrape remote sources;
- upload files or workflow artifacts;
- parse tensor or GGUF payloads;
- convert checkpoints;
- create an export file; or
- serialize model weights.

The validator performs filesystem metadata checks only and does not modify any file.
All writers remain dry-run only.

## Milestone status

Milestone 5.3 is now complete. Both local safetensors and GGUF F32 weight writers are implemented, tested, and integrated into the export CLI. See [MILESTONE_5_3_EXPORT_CLOSURE.md](MILESTONE_5_3_EXPORT_CLOSURE.md) for details.
