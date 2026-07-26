# Milestone 5.3 — Local Safetensors Writer

**Status:** Implemented

## Objective

Implement the first real, local-only safetensors model-weight writer for Bharat checkpoints,
and integrate it into the export registry and CLI through an explicit safe real-write mode.

## Supported checkpoint input format

The writer supports the **Bharat Model Format**:

- A **directory** containing `config.json` and `model.pt`, where `model.pt` is a raw `state_dict()`.
- A **`.pt` or `.pth` file** directly containing a state_dict.
- A **training checkpoint** (`.pt` file with a `"model"` key whose value is a state_dict).

The checkpoint is loaded with `torch.load(..., map_location="cpu", weights_only=True)`.
Non-dict top-level structures are rejected.

## Writer architecture

Module: `bharat/serving/safetensors_writer.py`

Public API:

```python
@dataclass(frozen=True)
class SafetensorsWriteResult:
    output_path: Path
    tensor_count: int
    bytes_written: int
    metadata: dict[str, str]

def write_safetensors_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    model_name: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> SafetensorsWriteResult: ...
```

## Registry integration

The `ExportWriterRegistry` now distinguishes between dry-run and real execution using
`(ExportFormat, bool)` keys. The default registry maps:

| Format | Dry-run | Writer |
|--------|---------|--------|
| safetensors | True | `safetensors-dry-run` |
| safetensors | False | `safetensors-local` |
| gguf | True | `gguf-dry-run` |
| gguf | False | (not implemented — raises error) |

A new adapter class `LocalSafetensorsExportWriter` wraps `write_safetensors_checkpoint()`
to conform to the `ExportWriter` protocol. It returns `ExportWriteResult` with `dry_run=False`
and the actual `bytes_written`.

## CLI integration: `--execute` flag

A new `--execute` flag controls real execution:

- **Without `--execute`**: existing dry-run behaviour is preserved.
- **With `--execute` and `--format safetensors`**: real safetensors file is written.
- **With `--execute` and `--format gguf`**: rejected with a clear error message.

When `--execute` is supplied:
- `ExportPlan.dry_run` is set to `False`.
- Checkpoint inventory is built automatically (required for writer readiness).
- Writer readiness runs automatically (validates output path, parent, etc.).
- The real safetensors writer creates the output file atomically.
- The manifest (if requested) is written only after the output succeeds.

### Execution ordering

1. Argument validation
2. Reject remote paths
3. Enforce format-specific rules
4. Build `ExportRequest` with `dry_run=False`
5. Build `ExportPlan`
6. Export path readiness (when path targets present)
7. Build checkpoint inventory (auto for `--execute`)
8. Safetensors metadata preflight (when requested)
9. Writer readiness (auto for `--execute`)
10. Manifest readiness (when manifest path supplied)
11. Real safetensors write via `LocalSafetensorsExportWriter`
12. Verify output exists and is non-empty
13. Write manifest (only after successful output)
14. Emit final JSON

### JSON output

**Dry-run:**
```json
{
  "dry_run": true,
  "writer_name": "safetensors-dry-run",
  "bytes_written": 0
}
```

**Real execution:**
```json
{
  "dry_run": false,
  "writer_name": "safetensors-local",
  "bytes_written": 12345,
  ...
}
```

## Command examples

**Safetensors dry-run:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local
```

**Safetensors real execution:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --execute
```

**Safetensors real execution with manifest:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --manifest-path ./exports/manifest.json \
    --execute
```

**Safetensors real execution with inventory:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.safetensors \
    --format safetensors \
    --model-name bharat-local \
    --include-inventory \
    --execute
```

**GGUF dry-run:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.gguf \
    --format gguf \
    --model-name bharat-local
```

**Rejected GGUF `--execute`:**
```bash
python -m scripts.run_export_plan \
    --checkpoint-path ./checkpoints/bharat \
    --output-path ./exports/bharat.gguf \
    --format gguf \
    --model-name bharat-local \
    --execute
# error: real GGUF export is not implemented; omit --execute for a dry-run
```

## Atomic-write behaviour

1. Create a uniquely named temporary file in the output directory via `tempfile.mkstemp`.
2. Write the safetensors payload to the temporary file.
3. Verify the temporary file exists and is non-empty.
4. Double-check that the output path does not exist.
5. Create a hard link from the temporary file to the output path via `os.link` (fails if output exists).
6. Unlink the temporary file.
7. If any step fails, remove the temporary file.

## No-overwrite guarantee

- The output path is checked for existence before writing begins.
- After writing to the temporary file, the output path is checked again.
- `os.link` fails with `FileExistsError` if the target already exists.
- The writer also rejects output paths inside the checkpoint directory.
- The CLI writer readiness check additionally prevents overwrites.

## Manifest transaction limitation

If the safetensors output file is created successfully but the manifest write fails:

- The output `.safetensors` file already exists and is valid.
- The manifest is not written.
- An error is reported to the user.
- The output file is not automatically deleted.

This is a deliberate design choice: a valid model weight file should not be silently removed
because of a secondary manifest failure. The user can re-run with a corrected manifest path.

## Dtype behaviour

Supported dtypes are those accepted by `safetensors.torch.save_file()`. Unsupported dtypes
raise `ValueError` with a clear message.

## Metadata

Stable metadata included in every safetensors file:

| Key | Value |
|-----|-------|
| `format` | `bharat-safetensors-v1` |
| `writer_version` | `1` |
| `model_name` | caller-provided (optional) |

Reserved keys cannot be overridden. All values are strings. Metadata is sorted deterministically.

## Safety boundaries

This writer does not:
- implement GGUF writing
- add quantization
- download models or datasets
- call external APIs
- require GPU
- accept arbitrary pickle objects
- weaken existing readiness validation
- remove dry-run support
- make real writing the default
- modify training, evaluation, tokenizer, API, data, or serving code

## Milestone status

Milestone 5.3 remains incomplete. GGUF export still does not exist.
The roadmap Export checkbox must remain unchecked.
