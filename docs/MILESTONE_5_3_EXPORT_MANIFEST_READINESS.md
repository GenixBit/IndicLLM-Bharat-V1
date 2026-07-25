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

## CLI integration

When `--manifest-path` is supplied, the CLI validates the manifest target before any export writer is invoked.

### Execution ordering

1. Build `ExportRequest` and `ExportPlan`.
2. Build checkpoint inventory when required.
3. Run format-specific metadata preflight (safetensors / GGUF).
4. Run writer readiness if `--validate-writer-readiness` is supplied.
5. Run manifest readiness validation.
6. Invoke dry-run export writer.
7. Construct and write the export manifest.
8. Emit final JSON output.

### Successful JSON output

```json
{
  "manifest_readiness": {
    "manifest_path": "/abs/path/to/manifest.json",
    "manifest_parent": "/abs/path/to",
    "output_path": "/abs/path/to/output.safetensors",
    "checkpoint_path": "/abs/path/to/checkpoint",
    "manifest_exists": false,
    "manifest_conflicts_with_output": false,
    "manifest_inside_checkpoint": false,
    "ready": true
  },
  "manifest_path": "/abs/path/to/manifest.json",
  "manifest_schema_version": "1.0"
}
```

### Failure behaviour

- Exits with a non-zero status.
- Prints a deterministic error message to stderr.
- Does not invoke the export writer.
- Does not create a manifest file.
- Does not create or modify the output file.
- Does not emit partial JSON to stdout.
- Does not leave temporary files.

### Protections

| Protection | Behaviour |
|-----------|-----------|
| No-overwrite | Rejects manifest path that already exists |
| Output collision | Rejects manifest path equal to output path |
| Checkpoint containment | Rejects manifest path inside checkpoint directory |
| Parent validation | Requires manifest parent to exist and be a directory |
| Path resolution | Uses `Path.resolve()` for `..`, relative, and symlink paths |

### Examples

**Safetensors with manifest:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --manifest-path ./manifests/bharat-export.json
```

**GGUF with manifest:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.gguf \
    --format gguf \
    --model-name bharat-local \
    --manifest-path ./manifests/bharat-export.json
```

**Combined writer and manifest readiness:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --validate-writer-readiness \
    --manifest-path ./manifests/bharat-export.json
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

The validator performs local filesystem metadata checks only and does not modify files.
All writers remain dry-run only.

## Milestone status

Milestone 5.3 remains incomplete. The repository still uses dry-run writers and does not contain real safetensors or GGUF model-weight serialization.
