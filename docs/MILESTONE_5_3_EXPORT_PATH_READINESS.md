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

## CLI integration

Export path readiness runs automatically when any of these CLI arguments is supplied:

- `--manifest-path`
- `--safetensors-metadata-path`
- `--gguf-metadata-path`

No separate CLI flag is needed.

### Execution ordering

1. Parse arguments.
2. Reject remote paths.
3. Enforce format-specific metadata rules.
4. Build `ExportRequest` and `ExportPlan`.
5. **Run export path readiness** (before inventory, preflight, or writer).
6. Build checkpoint inventory when needed.
7. Run safetensors / GGUF metadata preflight.
8. Run writer readiness when requested.
9. Run manifest readiness when manifest is supplied.
10. Invoke dry-run export writer.
11. Construct and write export manifest.
12. Emit final JSON.

### JSON output

```json
{
  "export_path_readiness": {
    "checkpoint_path": "/abs/path/to/checkpoint",
    "output_path": "/abs/path/to/output.safetensors",
    "manifest_path": "/abs/path/to/manifest.json",
    "metadata_paths": ["/abs/path/to/safetensors-meta.json"],
    "ready": true
  }
}
```

### Protections

| Protection | Behaviour |
|-----------|-----------|
| Duplicate metadata | Rejects identical resolved metadata paths |
| Metadata existence | Requires each metadata path to exist |
| Metadata type | Requires each metadata path to be a regular file |
| Output collision | Rejects metadata path equal to export output |
| Manifest collision | Rejects metadata path equal to manifest target |
| Path resolution | Uses `Path.resolve()` for `..`, relative, symlink paths |

### Failure behaviour

- Exits non-zero.
- Prints a deterministic error to stderr.
- Does not build checkpoint inventory.
- Does not read or parse metadata contents.
- Does not run safetensors or GGUF preflight.
- Does not run writer readiness.
- Does not run manifest readiness.
- Does not invoke the export writer.
- Does not create archive or manifest file.
- Does not emit partial JSON.
- Does not leave temporary files.

### Examples

**Safetensors with metadata:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --safetensors-metadata-path ./safetensors-metadata.json
```

**GGUF with metadata:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.gguf \
    --format gguf \
    --model-name bharat-local \
    --gguf-metadata-path ./gguf-metadata.json
```

**Manifest only:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --manifest-path ./manifests/bharat-export.json
```

**Combined readiness:**

```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --manifest-path ./manifests/bharat-export.json \
    --safetensors-metadata-path ./safetensors-metadata.json
```

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
All writers remain dry-run only.

## Milestone status

Milestone 5.3 remains incomplete. The repository still does not contain a real safetensors or GGUF weight writer. This gate reduces overwrite risk before a later approved local writer implementation.
